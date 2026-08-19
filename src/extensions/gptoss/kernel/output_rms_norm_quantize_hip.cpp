#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int hidden_size = 2880;
constexpr int block_size  = 1024;
constexpr int warp_size   = 32;
constexpr int q8_1_size   = 32;

struct block_q8_1 {
    __half2 ds;
    int8_t  qs[q8_1_size];
};

static_assert(sizeof(block_q8_1) == 36);

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, warp_size);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor(value, offset, warp_size));
    }
    return value;
}

__device__ float block_sum(float value, float * warp_sums) {
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;

    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    return warp_sum(warp_sums[lane]);
}

}  // namespace

__launch_bounds__(block_size) __global__ void gptoss_output_rms_norm_quantize_q8_1_kernel(
    const float * __restrict__ hidden,
    const float * __restrict__ weight,
    int32_t input_row,
    uint8_t * __restrict__ output,
    float eps) {
    __shared__ float warp_sums[block_size / warp_size];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;

    const float * input = hidden + static_cast<uint64_t>(input_row) * hidden_size;

    float sum = 0.0f;
    for (int column = threadIdx.x; column < hidden_size; column += block_size) {
        const float value = input[column];
        sum += value * value;
    }

    sum                   = block_sum(sum, warp_sums);
    const float inv_scale = rsqrtf(sum / hidden_size + eps);

    block_q8_1 * quantized = reinterpret_cast<block_q8_1 *>(output);

    for (int block = warp; block < hidden_size / q8_1_size; block += block_size / warp_size) {
        const int   column = block * q8_1_size + lane;
        const float value  = inv_scale * input[column] * weight[column];
        const float amax   = warp_max(fabsf(value));
        const float total  = warp_sum(value);
        const float scale  = amax / 127.0f;

        quantized[block].qs[lane] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));

        if (lane == 0) {
            quantized[block].ds = __floats2half2_rn(scale, total);
        }
    }
}
