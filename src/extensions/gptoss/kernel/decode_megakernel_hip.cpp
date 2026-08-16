// Faithful standalone port of TinyDwarfStar's BF16 decode megakernel.
//
// Control flow, reductions, barriers, RoPE scheduling, RMS recomputation, and
// MoE task ownership intentionally follow the current TDS source. The narrow
// adaptations are documented at their use sites: llama.cpp's SoA packed
// weights and F32 router tensors, FP16 activation materialization, and direct
// physical KV-row indirection.

#include "../gptoss-kernel.h"

#include <float.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace cg = cooperative_groups;

static constexpr uint32_t GPTOSS_EMBEDDING_LENGTH              = 2880;
static constexpr uint32_t GPTOSS_ATTENTION_HEAD_COUNT          = 64;
static constexpr uint32_t GPTOSS_ATTENTION_HEAD_COUNT_KV       = 8;
static constexpr uint32_t GPTOSS_ATTENTION_KEY_LENGTH          = 64;
static constexpr uint32_t GPTOSS_ATTENTION_VALUE_LENGTH        = 64;
static constexpr uint32_t GPTOSS_ATTENTION_SLIDING_WINDOW      = 128;
static constexpr uint32_t GPTOSS_EXPERT_COUNT                  = 32;
static constexpr uint32_t GPTOSS_EXPERT_USED_COUNT             = 4;
static constexpr uint32_t GPTOSS_EXPERT_FEED_FORWARD_LENGTH    = 2880;
static constexpr uint32_t GPTOSS_MEGA_ATTN_MAX_PARALLEL_BLOCKS = 12;

enum {
    TDS_MEGA_WARP            = 32,
    TDS_MEGA_WARPS_PER_BLOCK = 8,
    TDS_MEGA_BLOCK           = TDS_MEGA_WARP * TDS_MEGA_WARPS_PER_BLOCK,
    TDS_MEGA_RMS_BLOCKS      = TDS_MEGA_WARP / TDS_MEGA_WARPS_PER_BLOCK,
    TDS_MEGA_QK8_0           = 32,
    TDS_MEGA_QK8_1           = 32,
    TDS_MEGA_QK_MXFP4        = 32,
    TDS_MEGA_Q_DIM           = GPTOSS_ATTENTION_HEAD_COUNT * GPTOSS_ATTENTION_KEY_LENGTH,
    TDS_MEGA_KV_DIM          = GPTOSS_ATTENTION_HEAD_COUNT_KV * GPTOSS_ATTENTION_KEY_LENGTH,
};

static constexpr float TDS_MEGA_FATTN_KQ_MAX_OFFSET = 3.0f * 0.6931f;

using tds_mega_params = gptoss_decode_layer_params;

static constexpr uint32_t TDS_MEGA_QKV_ROWS                = TDS_MEGA_Q_DIM + 2u * TDS_MEGA_KV_DIM;
static constexpr uint32_t TDS_MEGA_MXFP4_PADDED_DIM        = 3072;
static constexpr uint32_t TDS_MEGA_MXFP4_VALUE_ROW_BYTES   = TDS_MEGA_MXFP4_PADDED_DIM / 2u;
static constexpr uint32_t TDS_MEGA_MXFP4_SCALE_ROW_BYTES   = TDS_MEGA_MXFP4_PADDED_DIM / TDS_MEGA_QK_MXFP4;
static constexpr uint64_t TDS_MEGA_QKV_VALUE_BYTES         = (uint64_t) TDS_MEGA_QKV_ROWS * GPTOSS_EMBEDDING_LENGTH;
static constexpr uint64_t TDS_MEGA_ATTN_OUTPUT_VALUE_BYTES = (uint64_t) GPTOSS_EMBEDDING_LENGTH * TDS_MEGA_Q_DIM;
static constexpr uint64_t TDS_MEGA_MOE_DOWN_VALUE_BYTES =
    (uint64_t) GPTOSS_EXPERT_COUNT * TDS_MEGA_MXFP4_PADDED_DIM * TDS_MEGA_MXFP4_VALUE_ROW_BYTES;
static constexpr uint64_t TDS_MEGA_MOE_GATE_UP_VALUE_BYTES = 2u * TDS_MEGA_MOE_DOWN_VALUE_BYTES;

struct tds_mega_q8_0_row {
    const int8_t * values;
    const half *   scales;
};

struct tds_mega_mxfp4_row {
    const uint8_t * values;
    const uint8_t * scales;
};

static __device__ __forceinline__ half * tds_mega_f16_norm(const tds_mega_params & p) {
    return p.activation_scratch;
}

static __device__ __forceinline__ half * tds_mega_f16_moe_act(const tds_mega_params & p) {
    return p.activation_scratch + GPTOSS_EMBEDDING_LENGTH;
}

static __device__ __forceinline__ float tds_mega_round_f16(float value) {
    return __half2float(__float2half_rn(value));
}

static __device__ __forceinline__ float tds_mega_warp_sum(float x) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int  row_mask   = 0xf;
    constexpr int  bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;
    x += __shfl_xor(x, 16, TDS_MEGA_WARP);
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x168, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x164, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x162, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x161, row_mask, bank_mask, bound_ctrl));
#else
#    pragma unroll
    for (int off = TDS_MEGA_WARP / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, TDS_MEGA_WARP);
    }
#endif
    return x;
}

static __device__ __forceinline__ float tds_mega_warp_sum8(float x) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int  row_mask   = 0xf;
    constexpr int  bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x164, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x162, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x161, row_mask, bank_mask, bound_ctrl));
#else
    x += __shfl_xor(x, 4, TDS_MEGA_WARP);
    x += __shfl_xor(x, 2, TDS_MEGA_WARP);
    x += __shfl_xor(x, 1, TDS_MEGA_WARP);
#endif
    return x;
}

static __device__ __forceinline__ float tds_mega_warp_max(float x) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int  row_mask   = 0xf;
    constexpr int  bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;
    x                         = fmaxf(x, __shfl_xor(x, 16, TDS_MEGA_WARP));
    x = fmaxf(x, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x168, row_mask,
                                                                    bank_mask, bound_ctrl)));
    x = fmaxf(x, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x164, row_mask,
                                                                    bank_mask, bound_ctrl)));
    x = fmaxf(x, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x162, row_mask,
                                                                    bank_mask, bound_ctrl)));
    x = fmaxf(x, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x161, row_mask,
                                                                    bank_mask, bound_ctrl)));
#else
#    pragma unroll
    for (int off = TDS_MEGA_WARP / 2; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_xor(x, off, TDS_MEGA_WARP));
    }
#endif
    return x;
}

static __device__ __forceinline__ float tds_mega_e8m0_to_fp32(uint8_t x) {
    const uint32_t bits = max((uint32_t) x << 23, 0x00400000u);
    return __uint_as_float(bits);
}

static __device__ __forceinline__ unsigned long long * tds_mega_atomic_topk_keys(const tds_mega_params & p) {
    return (unsigned long long *) (p.tmp + 32);
}

static __device__ __forceinline__ float * tds_mega_rms_partials(const tds_mega_params & p) {
    return p.tmp + 40;
}

static __device__ __forceinline__ uint32_t tds_mega_float_order_key(float x) {
    if (isnan(x)) {
        x = -FLT_MAX;
    }
    const uint32_t bits = __float_as_uint(x);
    const uint32_t mask = ((int32_t) bits < 0) ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}

static __device__ __forceinline__ unsigned long long tds_mega_topk_pack(float score, uint32_t expert) {
    const uint64_t ordered = (uint64_t) tds_mega_float_order_key(score);
    const uint64_t tie     = (uint64_t) (GPTOSS_EXPERT_COUNT - 1u - expert);
    return (unsigned long long) ((ordered << 32) | tie);
}

static __device__ __forceinline__ void tds_mega_atomic_topk_insert(const tds_mega_params & p,
                                                                   float                   score,
                                                                   uint32_t                expert) {
    unsigned long long   cur  = tds_mega_topk_pack(score, expert);
    unsigned long long * keys = tds_mega_atomic_topk_keys(p);
#pragma unroll
    for (uint32_t slot = 0; slot < GPTOSS_EXPERT_USED_COUNT; ++slot) {
        const unsigned long long old = atomicMax(keys + slot, cur);
        if (cur > old) {
            cur = old;
        }
        if (cur == 0ull) {
            break;  // The displaced value was an empty-slot sentinel.
        }
    }
}

static __device__ __forceinline__ tds_mega_q8_0_row
tds_mega_q8_0_row_at(const int8_t * values, const half * scales, uint32_t row, uint32_t columns, uint32_t blocks) {
    return {
        values + (uint64_t) row * columns,
        scales + (uint64_t) row * blocks,
    };
}

static __device__ __forceinline__ tds_mega_q8_0_row tds_mega_q8_0_row_offset(const tds_mega_q8_0_row & row,
                                                                             uint32_t                  offset,
                                                                             uint32_t                  columns,
                                                                             uint32_t                  blocks) {
    return {
        row.values + (uint64_t) offset * columns,
        row.scales + (uint64_t) offset * blocks,
    };
}

static __device__ __forceinline__ tds_mega_mxfp4_row tds_mega_mxfp4_row_at(const uint8_t * values,
                                                                           const uint8_t * scales,
                                                                           uint64_t        row) {
    return {
        values + row * TDS_MEGA_MXFP4_VALUE_ROW_BYTES,
        scales + row * TDS_MEGA_MXFP4_SCALE_ROW_BYTES,
    };
}

static __device__ __forceinline__ tds_mega_mxfp4_row tds_mega_mxfp4_row_offset(const tds_mega_mxfp4_row & row,
                                                                               uint64_t                   offset) {
    return {
        row.values + offset * TDS_MEGA_MXFP4_VALUE_ROW_BYTES,
        row.scales + offset * TDS_MEGA_MXFP4_SCALE_ROW_BYTES,
    };
}

static __device__ __forceinline__ uint32_t tds_mega_load_u32_unaligned(const void * x, int i32) {
    const uint8_t * x8 = (const uint8_t *) x;
    uint32_t        value;
    __builtin_memcpy(&value, x8 + 4 * i32, sizeof(value));
    return value;
}

using tds_mega_f16x2 = _Float16 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ void tds_mega_mad_f16x2(float & acc, half2 v, half2 u) {
#if defined(__HIP_PLATFORM_AMD__)
    acc = __builtin_amdgcn_fdot2(__builtin_bit_cast(tds_mega_f16x2, v), __builtin_bit_cast(tds_mega_f16x2, u), acc,
                                 false);
#else
    const float2 vf = __half22float2(v);
    const float2 uf = __half22float2(u);
    acc += vf.x * uf.x + vf.y * uf.y;
#endif
}

static __device__ __forceinline__ uint2 tds_mega_q8_to_f16x4(uint32_t packed) {
    packed ^= 0x80808080u;

    constexpr uint32_t exponent = 0x64646464u;
    const half2        offset   = __builtin_bit_cast(half2, 0xe480e480u);
    const half2        low  = __builtin_bit_cast(half2, __builtin_amdgcn_perm(packed, exponent, 0x00050004u)) + offset;
    const half2        high = __builtin_bit_cast(half2, __builtin_amdgcn_perm(packed, exponent, 0x00070006u)) + offset;

    return make_uint2(__builtin_bit_cast(uint32_t, low), __builtin_bit_cast(uint32_t, high));
}

static __device__ __forceinline__ float4 tds_mega_vec_dot4_q8_0_f16_same_x(const tds_mega_q8_0_row & row0,
                                                                           const tds_mega_q8_0_row & row1,
                                                                           const tds_mega_q8_0_row & row2,
                                                                           const tds_mega_q8_0_row & row3,
                                                                           const half * __restrict__ x,
                                                                           uint32_t block,
                                                                           int      kqs) {
    const uint32_t   first    = 4u * (uint32_t) kqs;
    const int8_t *   w0       = row0.values + (uint64_t) block * TDS_MEGA_QK8_0 + first;
    const int8_t *   w1       = row1.values + (uint64_t) block * TDS_MEGA_QK8_0 + first;
    const int8_t *   w2       = row2.values + (uint64_t) block * TDS_MEGA_QK8_0 + first;
    const int8_t *   w3       = row3.values + (uint64_t) block * TDS_MEGA_QK8_0 + first;
    const uint32_t * packed_x = (const uint32_t *) (x + (uint64_t) block * TDS_MEGA_QK8_0 + first);
    float            acc0     = 0.0f;
    float            acc1     = 0.0f;
    float            acc2     = 0.0f;
    float            acc3     = 0.0f;
#pragma unroll
    for (int word = 0; word < 2; ++word) {
        const uint32_t q0 = tds_mega_load_u32_unaligned(w0, word);
        const uint32_t q1 = tds_mega_load_u32_unaligned(w1, word);
        const uint32_t q2 = tds_mega_load_u32_unaligned(w2, word);
        const uint32_t q3 = tds_mega_load_u32_unaligned(w3, word);
        const uint2    v0 = tds_mega_q8_to_f16x4(q0);
        const uint2    v1 = tds_mega_q8_to_f16x4(q1);
        const uint2    v2 = tds_mega_q8_to_f16x4(q2);
        const uint2    v3 = tds_mega_q8_to_f16x4(q3);
        const half2    x0 = __builtin_bit_cast(half2, packed_x[2 * word]);
        const half2    x1 = __builtin_bit_cast(half2, packed_x[2 * word + 1]);
        tds_mega_mad_f16x2(acc0, __builtin_bit_cast(half2, v0.x), x0);
        tds_mega_mad_f16x2(acc0, __builtin_bit_cast(half2, v0.y), x1);
        tds_mega_mad_f16x2(acc1, __builtin_bit_cast(half2, v1.x), x0);
        tds_mega_mad_f16x2(acc1, __builtin_bit_cast(half2, v1.y), x1);
        tds_mega_mad_f16x2(acc2, __builtin_bit_cast(half2, v2.x), x0);
        tds_mega_mad_f16x2(acc2, __builtin_bit_cast(half2, v2.y), x1);
        tds_mega_mad_f16x2(acc3, __builtin_bit_cast(half2, v3.x), x0);
        tds_mega_mad_f16x2(acc3, __builtin_bit_cast(half2, v3.y), x1);
    }

    return make_float4(__half2float(row0.scales[block]) * acc0, __half2float(row1.scales[block]) * acc1,
                       __half2float(row2.scales[block]) * acc2, __half2float(row3.scales[block]) * acc3);
}

static __device__ __forceinline__ float4 tds_mega_reduce_dot4(float acc0, float acc1, float acc2, float acc3) {
    constexpr int    nwarps         = TDS_MEGA_WARPS_PER_BLOCK;
    constexpr int    partial_stride = (nwarps - 1) * TDS_MEGA_WARP;
    __shared__ float warp_partials[4 * partial_stride];
    const int        lane = threadIdx.x;
    const int        warp = threadIdx.y;

    if (warp > 0) {
        const int partial                           = (warp - 1) * TDS_MEGA_WARP + lane;
        warp_partials[0 * partial_stride + partial] = acc0;
        warp_partials[1 * partial_stride + partial] = acc1;
        warp_partials[2 * partial_stride + partial] = acc2;
        warp_partials[3 * partial_stride + partial] = acc3;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < nwarps - 1; ++w) {
            const int partial = w * TDS_MEGA_WARP + lane;
            acc0 += warp_partials[0 * partial_stride + partial];
            acc1 += warp_partials[1 * partial_stride + partial];
            acc2 += warp_partials[2 * partial_stride + partial];
            acc3 += warp_partials[3 * partial_stride + partial];
        }
        acc0 = tds_mega_warp_sum(acc0);
        acc1 = tds_mega_warp_sum(acc1);
        acc2 = tds_mega_warp_sum(acc2);
        acc3 = tds_mega_warp_sum(acc3);
    }
    __syncthreads();
    return make_float4(acc0, acc1, acc2, acc3);
}

static __device__ float4 tds_mega_dot4_q8_0_f16_block(const tds_mega_q8_0_row & row0,
                                                      const tds_mega_q8_0_row & row1,
                                                      const tds_mega_q8_0_row & row2,
                                                      const tds_mega_q8_0_row & row3,
                                                      const half * __restrict__ x,
                                                      uint32_t blocks) {
    constexpr int qi              = 8;
    constexpr int vdr             = 2;
    constexpr int nwarps          = TDS_MEGA_WARPS_PER_BLOCK;
    constexpr int blocks_per_iter = vdr * nwarps * TDS_MEGA_WARP / qi;
    const int     tid             = threadIdx.y * TDS_MEGA_WARP + threadIdx.x;
    float         acc0            = 0.0f;
    float         acc1            = 0.0f;
    float         acc2            = 0.0f;
    float         acc3            = 0.0f;
    const int     kqs             = vdr * (tid % (qi / vdr));
    for (uint32_t b = (uint32_t) (tid / (qi / vdr)); b < blocks; b += blocks_per_iter) {
        const float4 dot = tds_mega_vec_dot4_q8_0_f16_same_x(row0, row1, row2, row3, x, b, kqs);
        acc0 += dot.x;
        acc1 += dot.y;
        acc2 += dot.z;
        acc3 += dot.w;
    }

    return tds_mega_reduce_dot4(acc0, acc1, acc2, acc3);
}

static __device__ __forceinline__ uint4 tds_mega_fp4x8_to_f16x8(uint32_t codes) {
    constexpr uint32_t high0     = 0x3e3c3800u;
    constexpr uint32_t high1     = 0x46444240u;
    const uint32_t     magnitude = codes & 0x07070707u;
    const uint32_t     sign      = (codes & 0x08080808u) << 4;
    const uint32_t     high      = __builtin_amdgcn_perm(high1, high0, magnitude) | sign;
    const uint2        low =
        make_uint2(__builtin_amdgcn_perm(high, 0u, 0x05010400u), __builtin_amdgcn_perm(high, 0u, 0x07030602u));

    const uint32_t high_codes     = codes >> 4;
    const uint32_t high_magnitude = high_codes & 0x07070707u;
    const uint32_t high_sign      = (high_codes & 0x08080808u) << 4;
    const uint32_t high_values    = __builtin_amdgcn_perm(high1, high0, high_magnitude) | high_sign;
    const uint2    upper          = make_uint2(__builtin_amdgcn_perm(high_values, 0u, 0x05010400u),
                                               __builtin_amdgcn_perm(high_values, 0u, 0x07030602u));

    return make_uint4((low.x & 0x0000ffffu) | (upper.x << 16), (low.x >> 16) | (upper.x & 0xffff0000u),
                      (low.y & 0x0000ffffu) | (upper.y << 16), (low.y >> 16) | (upper.y & 0xffff0000u));
}

static __device__ __forceinline__ float tds_mega_vec_dot_mxfp4_f16(const tds_mega_mxfp4_row & row,
                                                                   const half * __restrict__ x,
                                                                   uint32_t block,
                                                                   int      kqs) {
    const uint8_t *  values  = row.values + (uint64_t) block * (TDS_MEGA_QK_MXFP4 / 2u);
    const uint32_t   segment = (uint32_t) kqs / 2u;
    const uint2      packed  = *(const uint2 *) (values + 8u * segment);
    const uint4      values0 = tds_mega_fp4x8_to_f16x8(packed.x);
    const uint4      values1 = tds_mega_fp4x8_to_f16x8(packed.y);
    const uint32_t * x2      = (const uint32_t *) (x + (uint64_t) block * TDS_MEGA_QK_MXFP4 + 16u * segment);
    float            sum     = 0.0f;
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values0.x), __builtin_bit_cast(half2, x2[0]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values0.y), __builtin_bit_cast(half2, x2[1]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values0.z), __builtin_bit_cast(half2, x2[2]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values0.w), __builtin_bit_cast(half2, x2[3]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values1.x), __builtin_bit_cast(half2, x2[4]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values1.y), __builtin_bit_cast(half2, x2[5]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values1.z), __builtin_bit_cast(half2, x2[6]));
    tds_mega_mad_f16x2(sum, __builtin_bit_cast(half2, values1.w), __builtin_bit_cast(half2, x2[7]));
    return tds_mega_e8m0_to_fp32(row.scales[block]) * sum;
}

static __device__ float2 tds_mega_dot2_mxfp4_f16_same_x_warp(const tds_mega_mxfp4_row & row0,
                                                             const tds_mega_mxfp4_row & row1,
                                                             const half * __restrict__ x) {
    constexpr uint32_t blocks          = GPTOSS_EXPERT_FEED_FORWARD_LENGTH / TDS_MEGA_QK_MXFP4;
    constexpr int      qi              = 4;
    constexpr int      vdr             = 2;
    constexpr int      blocks_per_iter = vdr * TDS_MEGA_WARP / qi;
    const int          lane            = threadIdx.x;
    float              acc0            = 0.0f;
    float              acc1            = 0.0f;
    const int          kqs             = vdr * (lane % (qi / vdr));
    for (uint32_t b = (uint32_t) (lane / (qi / vdr)); b < blocks; b += blocks_per_iter) {
        acc0 += tds_mega_vec_dot_mxfp4_f16(row0, x, b, kqs);
        acc1 += tds_mega_vec_dot_mxfp4_f16(row1, x, b, kqs);
    }
    return make_float2(tds_mega_warp_sum(acc0), tds_mega_warp_sum(acc1));
}

static __device__ float4 tds_mega_dot4_mxfp4_f16_same_x_warp(const tds_mega_mxfp4_row & row0,
                                                             const tds_mega_mxfp4_row & row1,
                                                             const tds_mega_mxfp4_row & row2,
                                                             const tds_mega_mxfp4_row & row3,
                                                             const half * __restrict__ x) {
    constexpr uint32_t blocks          = GPTOSS_EMBEDDING_LENGTH / TDS_MEGA_QK_MXFP4;
    constexpr int      qi              = 4;
    constexpr int      vdr             = 2;
    constexpr int      blocks_per_iter = vdr * TDS_MEGA_WARP / qi;
    const int          lane            = threadIdx.x;
    float              acc0            = 0.0f;
    float              acc1            = 0.0f;
    float              acc2            = 0.0f;
    float              acc3            = 0.0f;
    const int          kqs             = vdr * (lane % (qi / vdr));
    for (uint32_t b = (uint32_t) (lane / (qi / vdr)); b < blocks; b += blocks_per_iter) {
        acc0 += tds_mega_vec_dot_mxfp4_f16(row0, x, b, kqs);
        acc1 += tds_mega_vec_dot_mxfp4_f16(row1, x, b, kqs);
        acc2 += tds_mega_vec_dot_mxfp4_f16(row2, x, b, kqs);
        acc3 += tds_mega_vec_dot_mxfp4_f16(row3, x, b, kqs);
    }
    return make_float4(tds_mega_warp_sum(acc0), tds_mega_warp_sum(acc1), tds_mega_warp_sum(acc2),
                       tds_mega_warp_sum(acc3));
}

static __device__ __forceinline__ float tds_mega_swiglu_oai(float gate, float up) {
    gate            = fminf(gate, 7.0f);
    up              = fmaxf(fminf(up, 7.0f), -7.0f);
    const float e   = expf(-1.702f * gate);
    const float glu = gate / (1.0f + e);
    return glu * (1.0f + up);
}

static __device__ __forceinline__ float tds_mega_add_no_contract(float a, float b) {
#pragma clang fp                        contract(off)
    return a + b;
                       }

static __device__ __forceinline__ float tds_mega_mul_no_contract(float a, float b) {
#pragma clang fp                        contract(off)
    return a * b;
                       }

static __device__ void tds_mega_attention_rms_norm_f16(cg::grid_group grid, const tds_mega_params & p) {
    const uint32_t tid         = threadIdx.y * TDS_MEGA_WARP + threadIdx.x;
    const uint32_t logical_tid = blockIdx.x * TDS_MEGA_BLOCK + tid;

    const bool    reuse_previous_layer = p.reuse_attention_rms != 0u;
    const float * partials             = tds_mega_rms_partials(p);
    uint32_t      partial_count        = gridDim.x;
    if (!reuse_previous_layer) {
        float partial = 0.0f;
        if (blockIdx.x < TDS_MEGA_RMS_BLOCKS) {
            for (uint32_t i = logical_tid; i < GPTOSS_EMBEDDING_LENGTH;
                 i += (uint32_t) TDS_MEGA_RMS_BLOCKS * TDS_MEGA_BLOCK) {
                const float x = p.cur[i];
                partial += x * x;
            }
        }
        partial = tds_mega_warp_sum(partial);
        if (blockIdx.x < TDS_MEGA_RMS_BLOCKS && threadIdx.x == 0) {
            p.tmp[blockIdx.x * TDS_MEGA_WARPS_PER_BLOCK + threadIdx.y] = partial;
        }
        grid.sync();
        partials      = p.tmp;
        partial_count = TDS_MEGA_RMS_BLOCKS * TDS_MEGA_WARPS_PER_BLOCK;
    }

    __shared__ float block_scale;
    if (threadIdx.y == 0) {
        float total;
        if (reuse_previous_layer) {
            const uint32_t first = 4u * threadIdx.x;
            if (first < partial_count) {
                const float4 values = *(const float4 *) (partials + first);
                total               = values.x;
                if (first + 1u < partial_count) {
                    total += values.y;
                }
                if (first + 2u < partial_count) {
                    total += values.z;
                }
                if (first + 3u < partial_count) {
                    total += values.w;
                }
            } else {
                total = 0.0f;
            }
        } else {
            total = p.tmp[threadIdx.x];
        }
        total = tds_mega_warp_sum(total);
        if (threadIdx.x == 0) {
            block_scale = rsqrtf(total / (float) GPTOSS_EMBEDDING_LENGTH + p.rms_epsilon);
        }
    }
    __syncthreads();

    const float    scale        = block_scale;
    const uint32_t thread_count = gridDim.x * TDS_MEGA_BLOCK;
    for (uint32_t i = logical_tid; i < GPTOSS_EMBEDDING_LENGTH; i += thread_count) {
        tds_mega_f16_norm(p)[i] = __float2half_rn(p.cur[i] * scale * p.attn_norm[i]);
    }
}

static __device__ float tds_mega_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ void tds_mega_rope_yarn(const float   theta_extrap,
                                          const float   freq_scale,
                                          const float   corr_low,
                                          const float   corr_high,
                                          const int64_t i0,
                                          const float   ext_factor,
                                          float         mscale,
                                          float &       cos_theta,
                                          float &       sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = tds_mega_rope_yarn_ramp(corr_low, corr_high, (int) i0) * ext_factor;
        theta          = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

static __device__ __forceinline__ void tds_mega_rope_pair(const tds_mega_params & p,
                                                          uint32_t                pair,
                                                          float                   x0,
                                                          float                   x1,
                                                          float &                 y0,
                                                          float &                 y1) {
    const int   i0         = 2 * (int) pair;
    const float theta_base = (float) p.position * powf(p.rope_theta_scale, i0 / 2.0f);
    float       c;
    float       s;
    tds_mega_rope_yarn(theta_base, p.rope_freq_scale, p.rope_corr_low, p.rope_corr_high, i0, p.rope_ext_factor,
                       p.rope_attn_factor, c, s);
    /* The FP16 extension materializes the projected Q/K pair before RoPE,
       while its RoPE factors and rotation arithmetic remain FP32. Keep TDS's
       inline per-task factor schedule; only the precision boundary changes. */
    x0 = tds_mega_round_f16(x0);
    x1 = tds_mega_round_f16(x1);
    y0 = fmaf(x0, c, -(x1 * s));
    y1 = fmaf(x1, c, x0 * s);
}

/* Fused QKV epilogue. The normalized hidden row is FP16 while SoA Q8_0
   weights are dequantized into FP32 dot accumulation. Q/K/V linear outputs
   and staged RoPE operations observe matching FP16 boundaries before Q
   and K/V are stored in the existing FP16 scratch/cache representation. */
static __device__ void tds_mega_qkv_rope_cache_f16(const tds_mega_params & p) {
    constexpr uint32_t pairs_per_head      = GPTOSS_ATTENTION_KEY_LENGTH / 2u;
    constexpr uint32_t pair_quads_per_head = pairs_per_head / 2u;
    constexpr uint32_t q_pair_quads        = GPTOSS_ATTENTION_HEAD_COUNT * pair_quads_per_head;
    constexpr uint32_t k_pair_quads        = GPTOSS_ATTENTION_HEAD_COUNT_KV * pair_quads_per_head;
    constexpr uint32_t v_quads             = TDS_MEGA_KV_DIM / 4u;
    constexpr uint32_t total_tasks         = q_pair_quads + k_pair_quads + v_quads;
    const uint32_t     in_blocks           = GPTOSS_EMBEDDING_LENGTH / TDS_MEGA_QK8_0;
    const half * __restrict__ qkv_scales   = (const half *) (p.qkv_values + TDS_MEGA_QKV_VALUE_BYTES);
    const uint64_t kv_base                 = (uint64_t) p.kv_write_row * TDS_MEGA_KV_DIM;
    half * __restrict__ q_scaled           = p.query;

    for (uint32_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        if (task < q_pair_quads + k_pair_quads) {
            const bool     is_q             = task < q_pair_quads;
            const uint32_t pair_quad        = is_q ? task : task - q_pair_quads;
            const uint32_t head             = pair_quad / pair_quads_per_head;
            const uint32_t pair0            = (pair_quad % pair_quads_per_head) * 2u;
            const uint32_t row0             = head * GPTOSS_ATTENTION_KEY_LENGTH + pair0;
            const uint32_t row_base         = is_q ? 0u : TDS_MEGA_Q_DIM;
            const float * __restrict__ bias = is_q ? p.attn_q_bias : p.attn_k_bias;

            const tds_mega_q8_0_row weight0 =
                tds_mega_q8_0_row_at(p.qkv_values, qkv_scales, row_base + row0, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const tds_mega_q8_0_row weight1 =
                tds_mega_q8_0_row_offset(weight0, pairs_per_head, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const tds_mega_q8_0_row weight2 = tds_mega_q8_0_row_offset(weight0, 1u, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const tds_mega_q8_0_row weight3 = tds_mega_q8_0_row_offset(weight1, 1u, GPTOSS_EMBEDDING_LENGTH, in_blocks);

            const float4 dot =
                tds_mega_dot4_q8_0_f16_block(weight0, weight1, weight2, weight3, tds_mega_f16_norm(p), in_blocks);

            if (threadIdx.y == 0 && threadIdx.x < 2) {
                const uint32_t which = (uint32_t) threadIdx.x;
                const uint32_t pair  = pair0 + which;
                const uint32_t low   = row0 + which;
                const uint32_t high  = low + pairs_per_head;
                const float    x0    = (which == 0u ? dot.x : dot.z) + bias[low];
                const float    x1    = (which == 0u ? dot.y : dot.w) + bias[high];
                float          y0;
                float          y1;
                tds_mega_rope_pair(p, pair, x0, x1, y0, y1);
                if (is_q) {
                    q_scaled[low]  = __float2half_rn(y0 * 0.125f);
                    q_scaled[high] = __float2half_rn(y1 * 0.125f);
                } else {
                    p.cache_k[kv_base + low]  = __float2half_rn(y0);
                    p.cache_k[kv_base + high] = __float2half_rn(y1);
                }
            }
        } else {
            const uint32_t          v_quad     = task - q_pair_quads - k_pair_quads;
            const uint32_t          row        = v_quad * 4u;
            const uint32_t          weight_row = TDS_MEGA_Q_DIM + TDS_MEGA_KV_DIM + row;
            const tds_mega_q8_0_row weight0 =
                tds_mega_q8_0_row_at(p.qkv_values, qkv_scales, weight_row, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const tds_mega_q8_0_row weight1 = tds_mega_q8_0_row_offset(weight0, 1u, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const tds_mega_q8_0_row weight2 = tds_mega_q8_0_row_offset(weight1, 1u, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const tds_mega_q8_0_row weight3 = tds_mega_q8_0_row_offset(weight2, 1u, GPTOSS_EMBEDDING_LENGTH, in_blocks);
            const float4            dot =
                tds_mega_dot4_q8_0_f16_block(weight0, weight1, weight2, weight3, tds_mega_f16_norm(p), in_blocks);
            if (threadIdx.y == 0 && threadIdx.x == 0) {
                p.cache_v[kv_base + row] =
                    (half) tds_mega_round_f16(tds_mega_add_no_contract(dot.x, p.attn_v_bias[row]));
                p.cache_v[kv_base + row + 1u] =
                    (half) tds_mega_round_f16(tds_mega_add_no_contract(dot.y, p.attn_v_bias[row + 1u]));
                p.cache_v[kv_base + row + 2u] =
                    (half) tds_mega_round_f16(tds_mega_add_no_contract(dot.z, p.attn_v_bias[row + 2u]));
                p.cache_v[kv_base + row + 3u] =
                    (half) tds_mega_round_f16(tds_mega_add_no_contract(dot.w, p.attn_v_bias[row + 3u]));
            }
        }
    }
}

// The active window can intersect at most two globally aligned flash tiles.
// Keep both partials within one Q-head block, then combine them in the same
// partition order as standalone flash attention. Before the window fills it
// naturally covers the complete history.
static __device__ void tds_mega_window_attention_f16(const tds_mega_params & p) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t h    = blockIdx.x;
    if (h >= GPTOSS_ATTENTION_HEAD_COUNT) {
        return;
    }

    constexpr uint32_t tile_keys = 128u;
    static_assert(GPTOSS_ATTENTION_SLIDING_WINDOW == tile_keys, "SWA window must cover one flash tile");
    __shared__ float  scores[2u * tile_keys];
    __shared__ float2 tile_meta[2];

    const uint32_t gqa              = GPTOSS_ATTENTION_HEAD_COUNT / GPTOSS_ATTENTION_HEAD_COUNT_KV;
    const uint32_t hk               = h / gqa;
    const half *   qh               = p.query + (uint64_t) h * GPTOSS_ATTENTION_KEY_LENGTH;
    const uint32_t tid              = warp * TDS_MEGA_WARP + lane;
    const uint32_t history_count    = (uint32_t) p.position + 1u;
    const uint32_t window_count     = p.n_kv;
    const uint32_t window_start     = history_count - window_count;
    const uint32_t first_tile       = window_start / tile_keys;
    const uint32_t first_tile_begin = window_start % tile_keys;
    const uint32_t tile_span        = first_tile_begin + window_count;
    const uint32_t tile_count       = (tile_span + tile_keys - 1u) / tile_keys;
    for (uint32_t si = tid; si < window_count; si += TDS_MEGA_BLOCK) {
        const uint32_t key_in_window = first_tile_begin + si;
        const uint32_t tile          = key_in_window / tile_keys;
        const uint32_t key_in_tile   = key_in_window % tile_keys;
        const uint32_t row           = (uint32_t) p.kv_rows[si];
        const half *   kh  = p.cache_k + (uint64_t) row * TDS_MEGA_KV_DIM + (uint64_t) hk * GPTOSS_ATTENTION_KEY_LENGTH;
        float          dot = 0.0f;
#pragma unroll
        for (uint32_t d = 0; d < GPTOSS_ATTENTION_KEY_LENGTH; d += 2u) {
            tds_mega_mad_f16x2(dot, __halves2half2(kh[d], kh[d + 1u]), __halves2half2(qh[d], qh[d + 1u]));
        }
        scores[tile * tile_keys + key_in_tile] = dot;
    }
    __syncthreads();

    if (warp < tile_count) {
        const uint32_t begin      = warp == 0u ? first_tile_begin : 0u;
        const uint32_t tile_end   = tile_span - warp * tile_keys;
        const uint32_t end        = tile_end < tile_keys ? tile_end : tile_keys;
        const uint32_t score_base = warp * tile_keys;

        float kq_max = -FLT_MAX / 2.0f;
        for (uint32_t key_in_tile = lane; key_in_tile < tile_keys; key_in_tile += TDS_MEGA_WARP) {
            if (key_in_tile >= begin && key_in_tile < end) {
                kq_max = fmaxf(kq_max, scores[score_base + key_in_tile] + TDS_MEGA_FATTN_KQ_MAX_OFFSET);
            }
        }
        kq_max = tds_mega_warp_max(kq_max);

        float kq_sum = 0.0f;
        for (uint32_t key_in_tile = lane; key_in_tile < tile_keys; key_in_tile += TDS_MEGA_WARP) {
            if (key_in_tile >= begin && key_in_tile < end) {
                const float w                    = expf(scores[score_base + key_in_tile] - kq_max);
                scores[score_base + key_in_tile] = w;
                kq_sum += w;
            }
        }
        kq_sum = tds_mega_warp_sum(kq_sum);

        half2 value_acc = make_half2(0.0f, 0.0f);
#pragma unroll 8
        for (uint32_t key_in_tile = begin; key_in_tile < end; ++key_in_tile) {
            const uint32_t absolute_key = (first_tile + warp) * tile_keys + key_in_tile;
            const uint32_t logical_key  = absolute_key - window_start;
            const uint32_t row          = (uint32_t) p.kv_rows[logical_key];
            const half *   vh =
                p.cache_v + (uint64_t) row * TDS_MEGA_KV_DIM + (uint64_t) hk * GPTOSS_ATTENTION_VALUE_LENGTH;
            const float w = scores[score_base + key_in_tile];
            value_acc += __halves2half2(vh[lane], vh[lane + TDS_MEGA_WARP]) * make_half2(w, w);
        }

        const uint32_t part = (first_tile + warp) % p.attn_parallel_blocks;
        if (part == 0u) {
            const float sink       = p.attn_sinks[h];
            const float kq_max_new = fmaxf(kq_max, sink);
            const float scale      = expf(kq_max - kq_max_new);
            kq_max                 = kq_max_new;
            kq_sum                 = kq_sum * scale + expf(sink - kq_max);
            value_acc *= make_half2(scale, scale);
        }

        const float2 value                  = __half22float2(value_acc);
        scores[score_base + 2u * lane]      = value.x;
        scores[score_base + 2u * lane + 1u] = value.y;
        if (lane == 0u) {
            tile_meta[warp] = make_float2(kq_max, kq_sum);
        }
    }
    __syncthreads();

    float out0 = 0.0f;
    float out1 = 0.0f;
    if (warp == 0) {
        const uint32_t part0         = first_tile % p.attn_parallel_blocks;
        const uint32_t part1         = (first_tile + 1u) % p.attn_parallel_blocks;
        const bool     sink_has_tile = part0 == 0u || (tile_count == 2u && part1 == 0u);
        float          kq_max = sink_has_tile ? (part0 == 0u ? tile_meta[0].x : tile_meta[1].x) : p.attn_sinks[h];
        if (part0 != 0u) {
            kq_max = fmaxf(kq_max, tile_meta[0].x);
        }
        if (tile_count == 2u && part1 != 0u) {
            kq_max = fmaxf(kq_max, tile_meta[1].x);
        }

        float numerator0  = 0.0f;
        float numerator1  = 0.0f;
        float denominator = 0.0f;
        for (uint32_t part = 0; part < p.attn_parallel_blocks; ++part) {
            float  meta_max;
            float  meta_sum;
            float2 value;
            if (part == part0) {
                meta_max = tile_meta[0].x;
                meta_sum = tile_meta[0].y;
                value    = make_float2(scores[2u * lane], scores[2u * lane + 1u]);
            } else if (tile_count == 2u && part == part1) {
                meta_max = tile_meta[1].x;
                meta_sum = tile_meta[1].y;
                value    = make_float2(scores[tile_keys + 2u * lane], scores[tile_keys + 2u * lane + 1u]);
            } else if (part == 0u) {
                meta_max = p.attn_sinks[h];
                meta_sum = 1.0f;
                value    = make_float2(0.0f, 0.0f);
            } else {
                continue;
            }
            const float scale = expf(meta_max - kq_max);
            numerator0 += scale * value.x;
            numerator1 += scale * value.y;
            denominator += scale * meta_sum;
        }
        out0 = numerator0 / denominator;
        out1 = numerator1 / denominator;
    }

    if (warp == 0) {
        half * out                = p.query + (uint64_t) h * GPTOSS_ATTENTION_VALUE_LENGTH;
        out[lane]                 = __float2half_rn(out0);
        out[lane + TDS_MEGA_WARP] = __float2half_rn(out1);
    }
}

// Full causal decode attention inside the cooperative layer launch. The task
// layout matches flash_attn_tile: one task owns a KV head and one interleaved
// 128-key partition, while its eight warps own the eight GQA query heads. K and
// V are streamed through 32-key chunks so the layer kernel does not inherit the
// standalone flash kernel's 20+ KiB shared tile.
static __device__ void tds_mega_full_context_attention_parts(const half * __restrict__ q_f16,
                                                             const half * __restrict__ cache_k,
                                                             const half * __restrict__ cache_v,
                                                             const int32_t * __restrict__ kv_rows,
                                                             const float * __restrict__ attn_sinks,
                                                             float * __restrict__ attn_parts,
                                                             float2 * __restrict__ attn_meta,
                                                             uint32_t parallel_blocks,
                                                             uint32_t active_count) {
    constexpr uint32_t tile_keys        = 128u;
    constexpr uint32_t chunk_keys       = 32u;
    // A 64-half row is exactly 128 bytes and maps the same dimension of all
    // 32 keys to one LDS bank. Two pad halves make the half2 row stride 33
    // banks, rotating successive keys across every bank.
    constexpr uint32_t kv_shared_stride = GPTOSS_ATTENTION_KEY_LENGTH + 2u;
    constexpr uint32_t gqa              = GPTOSS_ATTENTION_HEAD_COUNT / GPTOSS_ATTENTION_HEAD_COUNT_KV;
    static_assert(gqa == TDS_MEGA_WARPS_PER_BLOCK, "one warp must own each GQA query head");
    static_assert(GPTOSS_ATTENTION_KEY_LENGTH == 64u && GPTOSS_ATTENTION_VALUE_LENGTH == 64u,
                  "streamed full attention is specialized to D=64");
    static_assert(chunk_keys * (GPTOSS_ATTENTION_KEY_LENGTH / 8u) == TDS_MEGA_BLOCK,
                  "one thread must copy one 16-byte KV segment");

    __shared__ half kv_chunk[2u * chunk_keys * kv_shared_stride];
    __shared__ half weights[TDS_MEGA_WARPS_PER_BLOCK * tile_keys];

    const uint32_t lane        = threadIdx.x;
    const uint32_t warp        = threadIdx.y;
    const uint32_t tid         = warp * TDS_MEGA_WARP + lane;
    const uint32_t task_count  = GPTOSS_ATTENTION_HEAD_COUNT_KV * parallel_blocks;
    const uint32_t tile_stride = parallel_blocks * tile_keys;

    for (uint32_t task = blockIdx.x; task < task_count; task += gridDim.x) {
        const uint32_t kv_head  = task / parallel_blocks;
        const uint32_t part     = task - kv_head * parallel_blocks;
        const uint32_t head     = kv_head * gqa + warp;
        const half *   q_head   = q_f16 + (uint64_t) head * GPTOSS_ATTENTION_KEY_LENGTH;
        half2 *        q_shared = (half2 *) (weights + warp * tile_keys);

        float kq_max    = -FLT_MAX / 2.0f;
        float kq_sum    = 0.0f;
        half2 value_acc = make_half2(0.0f, 0.0f);

        uint32_t tile = part * tile_keys;
        for (; tile < active_count; tile += tile_stride) {
            /* Softmax weights reuse this LDS row, so restore Q before every
               interleaved tile owned by the partition. */
            q_shared[lane] = *(const half2 *) (q_head + 2u * lane);
            __syncthreads();

            float          scores[tile_keys / TDS_MEGA_WARP];
            const uint32_t valid_chunks = (min(tile_keys, active_count - tile) + chunk_keys - 1u) / chunk_keys;

            const uint32_t copy_key  = tid / (GPTOSS_ATTENTION_KEY_LENGTH / 8u);
            const uint32_t copy_d8   = tid - copy_key * (GPTOSS_ATTENTION_KEY_LENGTH / 8u);
            const uint32_t first_key = tile + copy_key;
            uint4          first_k   = make_uint4(0u, 0u, 0u, 0u);
            if (first_key < active_count) {
                const uint32_t row = (uint32_t) kv_rows[first_key];
                first_k            = *(const uint4 *) (cache_k + (uint64_t) row * TDS_MEGA_KV_DIM +
                                            (uint64_t) kv_head * GPTOSS_ATTENTION_KEY_LENGTH + 8u * copy_d8);
            }
            uint32_t * first_k_dst = (uint32_t *) (kv_chunk + copy_key * kv_shared_stride + 8u * copy_d8);
            first_k_dst[0]         = first_k.x;
            first_k_dst[1]         = first_k.y;
            first_k_dst[2]         = first_k.z;
            first_k_dst[3]         = first_k.w;
            __syncthreads();

#pragma unroll 1
            for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
                const bool has_next = chunk + 1u < valid_chunks;
                uint4      next_k   = make_uint4(0u, 0u, 0u, 0u);
                if (has_next) {
                    const uint32_t next_key = tile + (chunk + 1u) * chunk_keys + copy_key;
                    if (next_key < active_count) {
                        const uint32_t row = (uint32_t) kv_rows[next_key];
                        next_k             = *(const uint4 *) (cache_k + (uint64_t) row * TDS_MEGA_KV_DIM +
                                                   (uint64_t) kv_head * GPTOSS_ATTENTION_KEY_LENGTH + 8u * copy_d8);
                    }
                }

                const half * current_k = kv_chunk + (chunk % 2u) * chunk_keys * kv_shared_stride;

                float dot0 = 0.0f;
#pragma unroll
                for (uint32_t d = 0; d < GPTOSS_ATTENTION_KEY_LENGTH; d += 8u) {
                    const half2 q0  = q_shared[d / 2u];
                    const half2 q1  = q_shared[d / 2u + 1u];
                    const half2 q2  = q_shared[d / 2u + 2u];
                    const half2 q3  = q_shared[d / 2u + 3u];
                    const half2 k00 = *(const half2 *) (current_k + lane * kv_shared_stride + d);
                    const half2 k01 = *(const half2 *) (current_k + lane * kv_shared_stride + d + 2u);
                    const half2 k02 = *(const half2 *) (current_k + lane * kv_shared_stride + d + 4u);
                    const half2 k03 = *(const half2 *) (current_k + lane * kv_shared_stride + d + 6u);
                    tds_mega_mad_f16x2(dot0, k00, q0);
                    tds_mega_mad_f16x2(dot0, k01, q1);
                    tds_mega_mad_f16x2(dot0, k02, q2);
                    tds_mega_mad_f16x2(dot0, k03, q3);
                }
                scores[chunk] = dot0;

                if (has_next) {
                    uint32_t * next_k_dst =
                        (uint32_t *) (kv_chunk + ((chunk + 1u) % 2u) * chunk_keys * kv_shared_stride +
                                      copy_key * kv_shared_stride + 8u * copy_d8);
                    next_k_dst[0] = next_k.x;
                    next_k_dst[1] = next_k.y;
                    next_k_dst[2] = next_k.z;
                    next_k_dst[3] = next_k.w;
                }
                __syncthreads();
            }

            float kq_max_new = kq_max;
#pragma unroll
            for (uint32_t i = 0; i < tile_keys / TDS_MEGA_WARP; ++i) {
                const uint32_t key = tile + i * TDS_MEGA_WARP + lane;
                if (key < active_count) {
                    kq_max_new = fmaxf(kq_max_new, scores[i] + TDS_MEGA_FATTN_KQ_MAX_OFFSET);
                }
            }
            kq_max_new = tds_mega_warp_max(kq_max_new);

            const float kq_max_scale = expf(kq_max - kq_max_new);
            kq_max                   = kq_max_new;
            float kq_sum_add         = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < tile_keys / TDS_MEGA_WARP; ++i) {
                const uint32_t key_in_tile = i * TDS_MEGA_WARP + lane;
                const uint32_t key         = tile + key_in_tile;
                const float    w           = key < active_count ? expf(scores[i] - kq_max) : 0.0f;
                kq_sum_add += w;
                weights[warp * tile_keys + key_in_tile] = (half) w;
            }
            kq_sum                  = kq_sum * kq_max_scale + kq_sum_add;
            const half2 value_scale = make_half2(kq_max_scale, kq_max_scale);
            __syncthreads();

            uint4 first_v = make_uint4(0u, 0u, 0u, 0u);
            if (first_key < active_count) {
                const uint32_t row = (uint32_t) kv_rows[first_key];
                first_v            = *(const uint4 *) (cache_v + (uint64_t) row * TDS_MEGA_KV_DIM +
                                            (uint64_t) kv_head * GPTOSS_ATTENTION_VALUE_LENGTH + 8u * copy_d8);
            }
            uint32_t * first_v_dst = (uint32_t *) (kv_chunk + copy_key * kv_shared_stride + 8u * copy_d8);
            first_v_dst[0]         = first_v.x;
            first_v_dst[1]         = first_v.y;
            first_v_dst[2]         = first_v.z;
            first_v_dst[3]         = first_v.w;
            __syncthreads();

#pragma unroll 1
            for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
                const bool has_next = chunk + 1u < valid_chunks;
                uint4      next_v   = make_uint4(0u, 0u, 0u, 0u);
                if (has_next) {
                    const uint32_t next_key = tile + (chunk + 1u) * chunk_keys + copy_key;
                    if (next_key < active_count) {
                        const uint32_t row = (uint32_t) kv_rows[next_key];
                        next_v             = *(const uint4 *) (cache_v + (uint64_t) row * TDS_MEGA_KV_DIM +
                                                   (uint64_t) kv_head * GPTOSS_ATTENTION_VALUE_LENGTH + 8u * copy_d8);
                    }
                }
                const half * current_v = kv_chunk + (chunk % 2u) * chunk_keys * kv_shared_stride;

#pragma unroll 1
                for (uint32_t key = 0; key < chunk_keys; key += 4u) {
                    const half2 v0 = *(const half2 *) (current_v + key * kv_shared_stride + 2u * lane);
                    const half2 v1 = *(const half2 *) (current_v + (key + 1u) * kv_shared_stride + 2u * lane);
                    const half2 v2 = *(const half2 *) (current_v + (key + 2u) * kv_shared_stride + 2u * lane);
                    const half2 v3 = *(const half2 *) (current_v + (key + 3u) * kv_shared_stride + 2u * lane);
                    const half  w0 = weights[warp * tile_keys + chunk * chunk_keys + key];
                    const half  w1 = weights[warp * tile_keys + chunk * chunk_keys + key + 1u];
                    const half  w2 = weights[warp * tile_keys + chunk * chunk_keys + key + 2u];
                    const half  w3 = weights[warp * tile_keys + chunk * chunk_keys + key + 3u];
                    if (chunk == 0u && key == 0u) {
                        /* Preserve standalone flash attention's separate first
                           multiply and following FP16 FMA rounding. */
                        const half2 first_product = __hmul2_rn(v0, __half2half2(w0));
                        value_acc                 = __hfma2(value_acc, value_scale, first_product);
                    } else {
                        value_acc += v0 * __half2half2(w0);
                    }
                    value_acc += v1 * __half2half2(w1);
                    value_acc += v2 * __half2half2(w2);
                    value_acc += v3 * __half2half2(w3);
                }

                if (has_next) {
                    uint32_t * next_v_dst =
                        (uint32_t *) (kv_chunk + ((chunk + 1u) % 2u) * chunk_keys * kv_shared_stride +
                                      copy_key * kv_shared_stride + 8u * copy_d8);
                    next_v_dst[0] = next_v.x;
                    next_v_dst[1] = next_v.y;
                    next_v_dst[2] = next_v.z;
                    next_v_dst[3] = next_v.w;
                }
                __syncthreads();
            }
        }

        kq_sum = tds_mega_warp_sum(kq_sum);
        if (part == 0u) {
            const float sink         = attn_sinks[head];
            const float kq_max_new   = fmaxf(kq_max, sink);
            const float kq_max_scale = expf(kq_max - kq_max_new);
            kq_max                   = kq_max_new;
            kq_sum                   = kq_sum * kq_max_scale + expf(sink - kq_max);
            value_acc *= make_half2(kq_max_scale, kq_max_scale);
        }

        const uint64_t part_index              = (uint64_t) head * parallel_blocks + part;
        const uint64_t part_base               = part_index * GPTOSS_ATTENTION_VALUE_LENGTH;
        const float2   value                   = __half22float2(value_acc);
        attn_parts[part_base + 2u * lane]      = value.x;
        attn_parts[part_base + 2u * lane + 1u] = value.y;
        if (lane == 0u) {
            attn_meta[part_index] = make_float2(kq_max, kq_sum);
        }
        __syncthreads();
    }
}

static __device__ void tds_mega_wo_quad4(const tds_mega_params & p) {
    const uint32_t in_blocks         = TDS_MEGA_Q_DIM / TDS_MEGA_QK8_0;
    const uint32_t total_quads       = GPTOSS_EMBEDDING_LENGTH >> 2;
    const half * __restrict__ scales = (const half *) (p.attn_output_values + TDS_MEGA_ATTN_OUTPUT_VALUE_BYTES);
    float ffn_rms_partial            = 0.0f;
    for (uint32_t quad = blockIdx.x; quad < total_quads; quad += gridDim.x) {
        const uint32_t          row = quad << 2;
        const tds_mega_q8_0_row row0 =
            tds_mega_q8_0_row_at(p.attn_output_values, scales, row, TDS_MEGA_Q_DIM, in_blocks);
        const tds_mega_q8_0_row row1 = tds_mega_q8_0_row_offset(row0, 1u, TDS_MEGA_Q_DIM, in_blocks);
        const tds_mega_q8_0_row row2 = tds_mega_q8_0_row_offset(row1, 1u, TDS_MEGA_Q_DIM, in_blocks);
        const tds_mega_q8_0_row row3 = tds_mega_q8_0_row_offset(row2, 1u, TDS_MEGA_Q_DIM, in_blocks);
        const float4            dot  = tds_mega_dot4_q8_0_f16_block(row0, row1, row2, row3, p.query, in_blocks);
        if (threadIdx.y == 0 && threadIdx.x == 0) {
            const float next0 = (dot.x + p.attn_output_bias[row]) + p.cur[row];
            const float next1 = (dot.y + p.attn_output_bias[row + 1u]) + p.cur[row + 1u];
            const float next2 = (dot.z + p.attn_output_bias[row + 2u]) + p.cur[row + 2u];
            const float next3 = (dot.w + p.attn_output_bias[row + 3u]) + p.cur[row + 3u];
            p.next[row]       = next0;
            p.next[row + 1u]  = next1;
            p.next[row + 2u]  = next2;
            p.next[row + 3u]  = next3;
            ffn_rms_partial += next0 * next0;
            ffn_rms_partial += next1 * next1;
            ffn_rms_partial += next2 * next2;
            ffn_rms_partial += next3 * next3;
        }
    }
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        tds_mega_rms_partials(p)[blockIdx.x] = ffn_rms_partial;
    }
    if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x < GPTOSS_EXPERT_USED_COUNT) {
        tds_mega_atomic_topk_keys(p)[threadIdx.x] = 0ull;
    }
}

static __device__ void tds_mega_full_context_combine_f16(const tds_mega_params & p) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid  = threadIdx.y * TDS_MEGA_WARP + threadIdx.x;
    if (head >= GPTOSS_ATTENTION_HEAD_COUNT || tid >= GPTOSS_ATTENTION_VALUE_LENGTH) {
        return;
    }

    const uint32_t parallel_blocks = p.attn_parallel_blocks;
    const float *  parts           = p.attn_parts + (uint64_t) head * parallel_blocks * GPTOSS_ATTENTION_VALUE_LENGTH;
    const float2 * meta            = p.attn_meta + (uint64_t) head * parallel_blocks;

    float kqmax = meta[0].x;
    for (uint32_t part = 1; part < parallel_blocks; ++part) {
        kqmax = fmaxf(kqmax, meta[part].x);
    }

    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (uint32_t part = 0; part < parallel_blocks; ++part) {
        const float scale = expf(meta[part].x - kqmax);
        numerator += scale * parts[(uint64_t) part * GPTOSS_ATTENTION_VALUE_LENGTH + tid];
        denominator += scale * meta[part].y;
    }
    const float out = numerator / denominator;

    p.query[(uint64_t) head * GPTOSS_ATTENTION_VALUE_LENGTH + tid] = __float2half_rn(out);
}

static __device__ void tds_mega_rms_ffn_f16_router(const tds_mega_params & p) {
    const uint32_t tid         = threadIdx.y * TDS_MEGA_WARP + threadIdx.x;
    const uint32_t logical_tid = blockIdx.x * TDS_MEGA_BLOCK + tid;

    __shared__ float block_scale;
    if (threadIdx.y == 0) {
        float total = 0.0f;
        for (uint32_t i = threadIdx.x; i < gridDim.x; i += TDS_MEGA_WARP) {
            total += tds_mega_rms_partials(p)[i];
        }
        total = tds_mega_warp_sum(total);
        if (threadIdx.x == 0) {
            block_scale = rsqrtf(total / (float) GPTOSS_EMBEDDING_LENGTH + p.rms_epsilon);
        }
    }
    __syncthreads();

    const float    scale        = block_scale;
    const uint32_t thread_count = gridDim.x * TDS_MEGA_BLOCK;
    for (uint32_t i = logical_tid; i < GPTOSS_EMBEDDING_LENGTH; i += thread_count) {
        tds_mega_f16_norm(p)[i] = __float2half_rn(p.next[i] * scale * p.post_attention_norm[i]);
    }

    const uint32_t   ncols2 = GPTOSS_EMBEDDING_LENGTH / 2u;
    __shared__ float buf_iw[TDS_MEGA_WARP];
    for (uint32_t e = blockIdx.x; e < GPTOSS_EXPERT_COUNT; e += gridDim.x) {
        if (tid < TDS_MEGA_WARP) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();

        const float2 * w2  = (const float2 *) (p.router_weight + e * GPTOSS_EMBEDDING_LENGTH);
        float          acc = 0.0f;
        for (uint32_t c2 = tid; c2 < ncols2; c2 += TDS_MEGA_BLOCK) {
            const uint32_t i   = c2 * 2u;
            const float2   wv  = w2[c2];
            /* Router weights are f32 and consume the normalized values directly;
               BF16 materialization is only for the MXFP4-weight matvecs. */
            const float    av0 = p.next[i] * scale * p.post_attention_norm[i];
            const float    av1 = p.next[i + 1u] * scale * p.post_attention_norm[i + 1u];
            acc += wv.x * av0;
            acc += wv.y * av1;
        }
        acc = tds_mega_warp_sum(acc);
        if (threadIdx.x == 0) {
            buf_iw[threadIdx.y] = acc;
        }
        __syncthreads();

        if (tid < TDS_MEGA_WARP) {
            float s = buf_iw[tid];
            s       = tds_mega_warp_sum(s);
            if (tid == 0) {
                const float score = s + p.router_bias[e];
                p.router[e]       = score;
                tds_mega_atomic_topk_insert(p, score, e);
            }
        }
        if (gridDim.x < GPTOSS_EXPERT_COUNT) {
            __syncthreads();
        }
    }
}

static __device__ void tds_mega_topk_from_atomic_keys(const tds_mega_params & p,
                                                      int32_t * __restrict__ ids,
                                                      bool write_global_result) {
    if (threadIdx.y != 0) {
        return;
    }

    const int lane   = threadIdx.x;
    int32_t   expert = 0;
    float     out_w  = 0.0f;
    if (lane < GPTOSS_EXPERT_USED_COUNT) {
        const unsigned long long key = tds_mega_atomic_topk_keys(p)[lane];
        const uint32_t           tie = (uint32_t) key;
        expert                       = (int32_t) (GPTOSS_EXPERT_COUNT - 1u - tie);
        ids[lane]                    = expert;
        if (write_global_result) {
            p.expert_ids[lane] = expert;
            out_w              = p.router[expert];
        }
    }

    if (!write_global_result) {
        return;
    }
    float max_val = lane < GPTOSS_EXPERT_USED_COUNT ? out_w : -INFINITY;
    max_val       = tds_mega_warp_max(max_val);
    float val     = 0.0f;
    if (lane < GPTOSS_EXPERT_USED_COUNT) {
        val = expf(out_w - max_val);
    }
    const float sum = tds_mega_warp_sum(val);
    const float inv = 1.0f / sum;
    if (lane < GPTOSS_EXPERT_USED_COUNT) {
        p.expert_weights[lane] = val * inv;
    }
}

static __device__ __forceinline__ void tds_mega_moe_up_task_slot_qb(uint32_t   task,
                                                                    uint32_t   out_blocks,
                                                                    uint32_t & slot,
                                                                    uint32_t & qb) {
    if (task < out_blocks) {
        slot = 0u;
        qb   = task;
    } else if (task < 2u * out_blocks) {
        slot = 1u;
        qb   = task - out_blocks;
    } else if (task < 3u * out_blocks) {
        slot = 2u;
        qb   = task - 2u * out_blocks;
    } else {
        slot = 3u;
        qb   = task - 3u * out_blocks;
    }
}

static __device__ void tds_mega_moe_up_gate_f16(const tds_mega_params & p) {
    const uint32_t out_blocks           = GPTOSS_EXPERT_FEED_FORWARD_LENGTH / TDS_MEGA_QK8_1;
    const uint32_t total_blocks         = GPTOSS_EXPERT_USED_COUNT * out_blocks;
    const int      lane                 = threadIdx.x;
    const half * __restrict__ f16_x     = tds_mega_f16_norm(p);
    half * __restrict__ f16_act         = tds_mega_f16_moe_act(p);
    const uint8_t * __restrict__ scales = p.moe_gate_up_values + TDS_MEGA_MOE_GATE_UP_VALUE_BYTES;
    __shared__ int32_t ids[GPTOSS_EXPERT_USED_COUNT];
    __shared__ float   act[TDS_MEGA_QK8_1];

    if (blockIdx.x >= total_blocks) {
        return;
    }

    tds_mega_topk_from_atomic_keys(p, ids, blockIdx.x == 0);
    __syncthreads();

    for (uint32_t task = blockIdx.x; task < total_blocks; task += gridDim.x) {
        uint32_t slot;
        uint32_t qb;
        tds_mega_moe_up_task_slot_qb(task, out_blocks, slot, qb);

        const uint32_t expert              = (uint32_t) ids[slot];
        const uint32_t row_base            = qb * TDS_MEGA_QK8_1;
        const uint64_t logical_expert_row  = (uint64_t) expert * GPTOSS_EXPERT_FEED_FORWARD_LENGTH + row_base;
        const uint64_t physical_expert_row = (uint64_t) expert * 2u * TDS_MEGA_MXFP4_PADDED_DIM + 2u * row_base;
        const tds_mega_mxfp4_row gate_tile = tds_mega_mxfp4_row_at(p.moe_gate_up_values, scales, physical_expert_row);
        const tds_mega_mxfp4_row up_tile   = tds_mega_mxfp4_row_offset(gate_tile, 1u);
        const float * __restrict__ bias    = p.moe_gate_up_bias + 2u * logical_expert_row;

        for (uint32_t row_in_block = 2u * (uint32_t) threadIdx.y; row_in_block < TDS_MEGA_QK8_1;
             row_in_block += 2u * TDS_MEGA_WARPS_PER_BLOCK) {
            const tds_mega_mxfp4_row gate0 = tds_mega_mxfp4_row_offset(gate_tile, 2u * row_in_block);
            const tds_mega_mxfp4_row up0   = tds_mega_mxfp4_row_offset(up_tile, 2u * row_in_block);
            const tds_mega_mxfp4_row gate1 = tds_mega_mxfp4_row_offset(gate0, 2u);
            const tds_mega_mxfp4_row up1   = tds_mega_mxfp4_row_offset(up0, 2u);
            const float4             dots  = tds_mega_dot4_mxfp4_f16_same_x_warp(gate0, up0, gate1, up1, f16_x);
            if (lane == 0) {
                const float gate       = dots.x + bias[2u * row_in_block];
                const float up         = dots.y + bias[2u * row_in_block + 1u];
                act[row_in_block]      = tds_mega_swiglu_oai(gate, up);
                const float gate_next  = dots.z + bias[2u * row_in_block + 2u];
                const float up_next    = dots.w + bias[2u * row_in_block + 3u];
                act[row_in_block + 1u] = tds_mega_swiglu_oai(gate_next, up_next);
            }
        }
        __syncthreads();

        if (threadIdx.y == 0) {
            f16_act[slot * GPTOSS_EXPERT_FEED_FORWARD_LENGTH + row_base + lane] = __float2half_rn(act[lane]);
        }
        if (task + gridDim.x < total_blocks) {
            __syncthreads();
        }
    }
}

static __device__ void tds_mega_moe_down_f16(const tds_mega_params & p) {
    const uint32_t warp_global          = blockIdx.x * TDS_MEGA_WARPS_PER_BLOCK + threadIdx.y;
    const uint32_t warp_count           = gridDim.x * TDS_MEGA_WARPS_PER_BLOCK;
    const uint32_t in_blocks            = GPTOSS_EXPERT_FEED_FORWARD_LENGTH / TDS_MEGA_QK_MXFP4;
    const int      lane                 = threadIdx.x;
    const half * __restrict__ f16_act   = tds_mega_f16_moe_act(p);
    const uint8_t * __restrict__ scales = p.moe_down_values + TDS_MEGA_MOE_DOWN_VALUE_BYTES;
    float * __restrict__ next           = p.next;
    __shared__ float rms_warp_partials[TDS_MEGA_WARPS_PER_BLOCK];

    if (blockIdx.x * TDS_MEGA_WARPS_PER_BLOCK >= GPTOSS_EMBEDDING_LENGTH) {
        return;
    }

    const uint32_t           expert0     = (uint32_t) p.expert_ids[0];
    const uint32_t           expert1     = (uint32_t) p.expert_ids[1];
    const uint32_t           expert2     = (uint32_t) p.expert_ids[2];
    const uint32_t           expert3     = (uint32_t) p.expert_ids[3];
    const float              weight0     = p.expert_weights[0];
    const float              weight1     = p.expert_weights[1];
    const float              weight2     = p.expert_weights[2];
    const float              weight3     = p.expert_weights[3];
    const half *             activation0 = f16_act;
    const half *             activation1 = activation0 + GPTOSS_EXPERT_FEED_FORWARD_LENGTH;
    const half *             activation2 = activation1 + GPTOSS_EXPERT_FEED_FORWARD_LENGTH;
    const half *             activation3 = activation2 + GPTOSS_EXPERT_FEED_FORWARD_LENGTH;
    const tds_mega_mxfp4_row expert_row0 =
        tds_mega_mxfp4_row_at(p.moe_down_values, scales, (uint64_t) expert0 * TDS_MEGA_MXFP4_PADDED_DIM);
    const tds_mega_mxfp4_row expert_row1 =
        tds_mega_mxfp4_row_at(p.moe_down_values, scales, (uint64_t) expert1 * TDS_MEGA_MXFP4_PADDED_DIM);
    const tds_mega_mxfp4_row expert_row2 =
        tds_mega_mxfp4_row_at(p.moe_down_values, scales, (uint64_t) expert2 * TDS_MEGA_MXFP4_PADDED_DIM);
    const tds_mega_mxfp4_row expert_row3 =
        tds_mega_mxfp4_row_at(p.moe_down_values, scales, (uint64_t) expert3 * TDS_MEGA_MXFP4_PADDED_DIM);
    const float * bias0 = p.moe_down_bias + (uint64_t) expert0 * GPTOSS_EMBEDDING_LENGTH;
    const float * bias1 = p.moe_down_bias + (uint64_t) expert1 * GPTOSS_EMBEDDING_LENGTH;
    const float * bias2 = p.moe_down_bias + (uint64_t) expert2 * GPTOSS_EMBEDDING_LENGTH;
    const float * bias3 = p.moe_down_bias + (uint64_t) expert3 * GPTOSS_EMBEDDING_LENGTH;

    constexpr uint32_t row_pair_count = GPTOSS_EMBEDDING_LENGTH / 2u;
    float              rms_partial    = 0.0f;
    for (uint32_t pair = warp_global; pair < row_pair_count; pair += warp_count) {
        const uint32_t           row0  = 2u * pair;
        const tds_mega_mxfp4_row row00 = tds_mega_mxfp4_row_offset(expert_row0, row0);
        const tds_mega_mxfp4_row row10 = tds_mega_mxfp4_row_offset(expert_row1, row0);
        const tds_mega_mxfp4_row row20 = tds_mega_mxfp4_row_offset(expert_row2, row0);
        const tds_mega_mxfp4_row row30 = tds_mega_mxfp4_row_offset(expert_row3, row0);
        const float2             dot0 =
            tds_mega_dot2_mxfp4_f16_same_x_warp(row00, tds_mega_mxfp4_row_offset(row00, 1u), activation0);
        const float2 dot1 =
            tds_mega_dot2_mxfp4_f16_same_x_warp(row10, tds_mega_mxfp4_row_offset(row10, 1u), activation1);
        const float2 dot2 =
            tds_mega_dot2_mxfp4_f16_same_x_warp(row20, tds_mega_mxfp4_row_offset(row20, 1u), activation2);
        const float2 dot3 =
            tds_mega_dot2_mxfp4_f16_same_x_warp(row30, tds_mega_mxfp4_row_offset(row30, 1u), activation3);

        if (lane < 2) {
            const uint32_t output_row = row0 + (uint32_t) lane;
            const float    value0     = lane == 0 ? dot0.x : dot0.y;
            const float    value1     = lane == 0 ? dot1.x : dot1.y;
            const float    value2     = lane == 0 ? dot2.x : dot2.y;
            const float    value3     = lane == 0 ? dot3.x : dot3.y;
            float total = tds_mega_mul_no_contract(weight0, tds_mega_add_no_contract(value0, bias0[output_row]));
            total       = tds_mega_add_no_contract(
                total, tds_mega_mul_no_contract(weight1, tds_mega_add_no_contract(value1, bias1[output_row])));
            total = tds_mega_add_no_contract(
                total, tds_mega_mul_no_contract(weight2, tds_mega_add_no_contract(value2, bias2[output_row])));
            total = tds_mega_add_no_contract(
                total, tds_mega_mul_no_contract(weight3, tds_mega_add_no_contract(value3, bias3[output_row])));

            const float result = tds_mega_add_no_contract(next[output_row], total);
            next[output_row]   = result;
            rms_partial += result * result;
        }
    }

    rms_partial = tds_mega_warp_sum(rms_partial);
    if (lane == 0) {
        rms_warp_partials[threadIdx.y] = rms_partial;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        float block_partial = lane < TDS_MEGA_WARPS_PER_BLOCK ? rms_warp_partials[lane] : 0.0f;
        block_partial       = tds_mega_warp_sum8(block_partial);
        if (lane == 0) {
            tds_mega_rms_partials(p)[blockIdx.x] = block_partial;
        }
    }
}

// Complete sliding-window decode layer.
__launch_bounds__(TDS_MEGA_BLOCK, 1) __global__ void gptoss_decode_layer_swa_kernel(gptoss_decode_layer_params p) {
    cg::grid_group grid = cg::this_grid();

    tds_mega_attention_rms_norm_f16(grid, p);
    grid.sync();
    tds_mega_qkv_rope_cache_f16(p);
    grid.sync();
    tds_mega_window_attention_f16(p);
    grid.sync();
    tds_mega_wo_quad4(p);
    grid.sync();
    tds_mega_rms_ffn_f16_router(p);
    grid.sync();
    tds_mega_moe_up_gate_f16(p);
    grid.sync();
    tds_mega_moe_down_f16(p);
}

// Complete full-context decode layer. The producer writes flash-style partials
// for all heads and partitions; the cooperative grid combines directly to
// FP16 and continues through the shared output/MoE stages.
__launch_bounds__(TDS_MEGA_BLOCK, 1) __global__ void gptoss_decode_layer_full_kernel(gptoss_decode_layer_params p) {
    cg::grid_group grid = cg::this_grid();

    tds_mega_attention_rms_norm_f16(grid, p);
    grid.sync();
    tds_mega_qkv_rope_cache_f16(p);
    grid.sync();
    tds_mega_full_context_attention_parts(p.query, p.cache_k, p.cache_v, p.kv_rows, p.attn_sinks, p.attn_parts,
                                          p.attn_meta, p.attn_parallel_blocks, p.n_kv);
    grid.sync();
    tds_mega_full_context_combine_f16(p);
    grid.sync();
    tds_mega_wo_quad4(p);
    grid.sync();
    tds_mega_rms_ffn_f16_router(p);
    grid.sync();
    tds_mega_moe_up_gate_f16(p);
    grid.sync();
    tds_mega_moe_down_f16(p);
}
