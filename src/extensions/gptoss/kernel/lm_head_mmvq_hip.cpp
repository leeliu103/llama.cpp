#include "../gptoss-config.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int hidden_size           = gptoss_hidden_size;
constexpr int vocabulary_size       = gptoss_vocabulary_size;
constexpr int rows_per_block        = 4;
constexpr int block_size            = 128;
constexpr int warp_size             = 32;
constexpr int warp_count            = block_size / warp_size;
constexpr int quant_block_size      = gptoss_quant_block_size;
constexpr int blocks_per_row        = hidden_size / quant_block_size;
constexpr int segments_per_q8_block = 4;

static_assert(vocabulary_size % rows_per_block == 0);

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[quant_block_size];
};

static_assert(sizeof(block_q8_0) == 34);

__device__ __forceinline__ float warp_sum(float value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int  row_mask   = 0xf;
    constexpr int  bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;
    value += __shfl_xor(value, 16, warp_size);
    value += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, value), 0x168, row_mask, bank_mask, bound_ctrl));
    value += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, value), 0x164, row_mask, bank_mask, bound_ctrl));
    value += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, value), 0x162, row_mask, bank_mask, bound_ctrl));
    value += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, value), 0x161, row_mask, bank_mask, bound_ctrl));
#else
#    pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, warp_size);
    }
#endif
    return value;
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const void * data, int word) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint32_t        value;
    __builtin_memcpy(&value, bytes + 4 * word, sizeof(value));
    return value;
}

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void mad_f16x2(float & acc, __half2 v, __half2 x) {
    acc = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, v), __builtin_bit_cast(f16x2, x), acc, false);
}

__device__ __forceinline__ uint2 q8_to_f16x4(uint32_t packed) {
    packed ^= 0x80808080u;

    constexpr uint32_t exponent = 0x64646464u;
    const __half2      offset   = __builtin_bit_cast(__half2, 0xe480e480u);
    const __half2      low = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(packed, exponent, 0x00050004u)) + offset;
    const __half2 high     = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(packed, exponent, 0x00070006u)) + offset;

    return make_uint2(__builtin_bit_cast(uint32_t, low), __builtin_bit_cast(uint32_t, high));
}

__device__ __forceinline__ float4 q8_0_dot4_segment(const block_q8_0 & row0,
                                                    const block_q8_0 & row1,
                                                    const block_q8_0 & row2,
                                                    const block_q8_0 & row3,
                                                    const __half *     activation,
                                                    int                segment) {
    constexpr int    values_per_segment = quant_block_size / segments_per_q8_block;
    const int        value_offset       = segment * values_per_segment;
    const int8_t *   values0            = row0.qs + value_offset;
    const int8_t *   values1            = row1.qs + value_offset;
    const int8_t *   values2            = row2.qs + value_offset;
    const int8_t *   values3            = row3.qs + value_offset;
    const uint32_t * activation_pairs   = reinterpret_cast<const uint32_t *>(activation + value_offset);
    float            acc0               = 0.0f;
    float            acc1               = 0.0f;
    float            acc2               = 0.0f;
    float            acc3               = 0.0f;

#pragma unroll
    for (int word = 0; word < 2; ++word) {
        const uint2   q0 = q8_to_f16x4(load_u32_unaligned(values0, word));
        const uint2   q1 = q8_to_f16x4(load_u32_unaligned(values1, word));
        const uint2   q2 = q8_to_f16x4(load_u32_unaligned(values2, word));
        const uint2   q3 = q8_to_f16x4(load_u32_unaligned(values3, word));
        const __half2 x0 = __builtin_bit_cast(__half2, activation_pairs[2 * word]);
        const __half2 x1 = __builtin_bit_cast(__half2, activation_pairs[2 * word + 1]);
        mad_f16x2(acc0, __builtin_bit_cast(__half2, q0.x), x0);
        mad_f16x2(acc0, __builtin_bit_cast(__half2, q0.y), x1);
        mad_f16x2(acc1, __builtin_bit_cast(__half2, q1.x), x0);
        mad_f16x2(acc1, __builtin_bit_cast(__half2, q1.y), x1);
        mad_f16x2(acc2, __builtin_bit_cast(__half2, q2.x), x0);
        mad_f16x2(acc2, __builtin_bit_cast(__half2, q2.y), x1);
        mad_f16x2(acc3, __builtin_bit_cast(__half2, q3.x), x0);
        mad_f16x2(acc3, __builtin_bit_cast(__half2, q3.y), x1);
    }

    return make_float4(__half2float(__ushort_as_half(row0.d)) * acc0, __half2float(__ushort_as_half(row1.d)) * acc1,
                       __half2float(__ushort_as_half(row2.d)) * acc2, __half2float(__ushort_as_half(row3.d)) * acc3);
}

}  // namespace

__launch_bounds__(block_size) __global__ void gptoss_lm_head_mmvq_q8_0_f16_kernel(
    const uint8_t * __restrict__ weight,
    const __half * __restrict__ activation,
    float * __restrict__ logits) {
    __shared__ float warp_sums[rows_per_block][warp_count - 1][warp_size];

    const int          lane      = threadIdx.x;
    const int          warp      = threadIdx.y;
    const int          thread    = warp * warp_size + lane;
    const int          first_row = blockIdx.x * rows_per_block;
    const block_q8_0 * rows[rows_per_block];
#pragma unroll
    for (int row = 0; row < rows_per_block; ++row) {
        rows[row] =
            reinterpret_cast<const block_q8_0 *>(weight) + static_cast<uint64_t>(first_row + row) * blocks_per_row;
    }

    float4    sum          = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int segment      = thread % segments_per_q8_block;
    const int block_stride = block_size / segments_per_q8_block;
    for (int block = thread / segments_per_q8_block; block < blocks_per_row; block += block_stride) {
        const float4 dot = q8_0_dot4_segment(rows[0][block], rows[1][block], rows[2][block], rows[3][block],
                                             activation + block * quant_block_size, segment);
        sum.x += dot.x;
        sum.y += dot.y;
        sum.z += dot.z;
        sum.w += dot.w;
    }

    if (warp > 0) {
        warp_sums[0][warp - 1][lane] = sum.x;
        warp_sums[1][warp - 1][lane] = sum.y;
        warp_sums[2][warp - 1][lane] = sum.z;
        warp_sums[3][warp - 1][lane] = sum.w;
    }
    __syncthreads();

    if (warp > 0) {
        return;
    }

#pragma unroll
    for (int index = 0; index < warp_count - 1; ++index) {
        sum.x += warp_sums[0][index][lane];
        sum.y += warp_sums[1][index][lane];
        sum.z += warp_sums[2][index][lane];
        sum.w += warp_sums[3][index][lane];
    }

    sum.x = warp_sum(sum.x);
    sum.y = warp_sum(sum.y);
    sum.z = warp_sum(sum.z);
    sum.w = warp_sum(sum.w);

    if (lane == 0) {
        logits[first_row]     = sum.x;
        logits[first_row + 1] = sum.y;
        logits[first_row + 2] = sum.z;
        logits[first_row + 3] = sum.w;
    }
}
