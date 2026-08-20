#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int hidden_size = 2880;
constexpr int block_size  = 1024;
constexpr int warp_size   = 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, warp_size);
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

__launch_bounds__(block_size) __global__ void gptoss_rms_norm_f16_kernel(const float * __restrict__ input,
                                                                         const float * __restrict__ weight,
                                                                         __half * __restrict__ output,
                                                                         float eps) {
    __shared__ float warp_sums[block_size / warp_size];

    const std::uint64_t row_offset = static_cast<std::uint64_t>(blockIdx.x) * hidden_size;
    input += row_offset;
    output += row_offset;

    float sum = 0.0f;
    for (int column = threadIdx.x; column < hidden_size; column += block_size) {
        const float value = input[column];
        sum += value * value;
    }

    sum                   = block_sum(sum, warp_sums);
    const float inv_scale = rsqrtf(sum / hidden_size + eps);

    for (int column = threadIdx.x; column < hidden_size; column += block_size) {
        float value = inv_scale * input[column] * weight[column];
        asm volatile("" : "+v"(value));
        output[column] = __float2half_rn(value);
    }
}

__launch_bounds__(block_size) __global__ void gptoss_post_attention_rms_norm_f32_f16_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    float * __restrict__ output_f32,
    __half * __restrict__ output_f16,
    float eps) {
    __shared__ float warp_sums[block_size / warp_size];

    const std::uint64_t row_offset = static_cast<std::uint64_t>(blockIdx.x) * hidden_size;
    input += row_offset;
    output_f32 += row_offset;
    output_f16 += row_offset;

    float sum = 0.0f;
    for (int column = threadIdx.x; column < hidden_size; column += block_size) {
        const float value = input[column];
        sum += value * value;
    }

    sum                   = block_sum(sum, warp_sums);
    const float inv_scale = rsqrtf(sum / hidden_size + eps);

    for (int column = threadIdx.x; column < hidden_size; column += block_size) {
        float value = inv_scale * input[column] * weight[column];
        asm volatile("" : "+v"(value));
        output_f32[column] = value;
        output_f16[column] = __float2half_rn(value);
    }
}
