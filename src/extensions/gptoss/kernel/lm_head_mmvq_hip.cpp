#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int hidden_size      = 2880;
constexpr int block_size       = 256;
constexpr int warp_size        = 32;
constexpr int warp_count       = block_size / warp_size;
constexpr int quant_block_size = 32;
constexpr int blocks_per_row   = hidden_size / quant_block_size;

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[quant_block_size];
};

struct block_q8_1 {
    uint16_t d;
    uint16_t s;
    int8_t   qs[quant_block_size];
};

static_assert(sizeof(block_q8_0) == 34);
static_assert(sizeof(block_q8_1) == 36);

__device__ __forceinline__ int load_weight_values(const int8_t * values, int index) {
    const uint16_t * halves = reinterpret_cast<const uint16_t *>(values);
    const uint32_t   low    = halves[2 * index];
    const uint32_t   high   = halves[2 * index + 1];
    return static_cast<int>(low | high << 16);
}

__device__ __forceinline__ int dot4(int lhs, int rhs, int sum) {
#if defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sudot4(true, lhs, true, rhs, sum, false);
#else
    const int8_t * lhs_values = reinterpret_cast<const int8_t *>(&lhs);
    const int8_t * rhs_values = reinterpret_cast<const int8_t *>(&rhs);
    return sum + lhs_values[0] * rhs_values[0] + lhs_values[1] * rhs_values[1] + lhs_values[2] * rhs_values[2] +
           lhs_values[3] * rhs_values[3];
#endif
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, warp_size);
    }
    return value;
}

}  // namespace

__launch_bounds__(block_size) __global__ void gptoss_lm_head_mmvq_q8_0_kernel(const uint8_t * __restrict__ weight,
                                                                              const uint8_t * __restrict__ activation,
                                                                              float * __restrict__ logits) {
    __shared__ float warp_sums[warp_count - 1][warp_size];

    const int vocabulary_row = blockIdx.x;
    const int thread         = threadIdx.x;
    const int lane           = thread % warp_size;
    const int warp           = thread / warp_size;

    const block_q8_0 * weight_row =
        reinterpret_cast<const block_q8_0 *>(weight) + static_cast<uint64_t>(vocabulary_row) * blocks_per_row;
    const block_q8_1 * activation_row = reinterpret_cast<const block_q8_1 *>(activation);

    float sum = 0.0f;

    const int values_per_thread = 8;
    const int block_stride      = block_size / (quant_block_size / values_per_thread);
    const int value_group       = thread % (quant_block_size / values_per_thread);

    for (int block = thread / (quant_block_size / values_per_thread); block < blocks_per_row; block += block_stride) {
        const int   value_index       = 2 * value_group;
        const int * activation_values = reinterpret_cast<const int *>(activation_row[block].qs);

        int dot = 0;
        dot     = dot4(load_weight_values(weight_row[block].qs, value_index), activation_values[value_index], dot);
        dot = dot4(load_weight_values(weight_row[block].qs, value_index + 1), activation_values[value_index + 1], dot);

        const float weight_scale     = __half2float(__ushort_as_half(weight_row[block].d));
        const float activation_scale = __half2float(__ushort_as_half(activation_row[block].d));
        sum += weight_scale * activation_scale * dot;
    }

    if (warp > 0) {
        warp_sums[warp - 1][lane] = sum;
    }
    __syncthreads();

    if (warp > 0) {
        return;
    }

#pragma unroll
    for (int index = 0; index < warp_count - 1; ++index) {
        sum += warp_sums[index][lane];
    }

    sum = warp_sum(sum);

    if (lane == 0) {
        logits[vocabulary_row] = sum;
    }
}
