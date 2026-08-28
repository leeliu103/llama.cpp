// GPT-OSS FP16 decode megakernel. Each cooperative launch executes one
// transformer layer using llama.cpp's packed weights and physical KV rows.

#include "../gptoss-config.h"
#include "../gptoss-kernel-hip.h"

#include <float.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

static constexpr uint32_t hidden_size         = gptoss_hidden_size;
static constexpr uint32_t query_head_count    = gptoss_query_head_count;
static constexpr uint32_t kv_head_count       = gptoss_kv_head_count;
static constexpr uint32_t head_size           = gptoss_head_size;
static constexpr uint32_t sliding_window_size = gptoss_swa_size;
static constexpr uint32_t expert_count        = gptoss_expert_count;
static constexpr uint32_t experts_used        = gptoss_expert_used_count;
static constexpr uint32_t intermediate_size   = gptoss_intermediate_size;

static constexpr uint32_t warp_size            = gptoss_decode_block_x;
static constexpr uint32_t warps_per_block      = gptoss_decode_block_y;
static constexpr uint32_t block_size           = warp_size * warps_per_block;
static constexpr uint32_t q8_block_size        = gptoss_quant_block_size;
static constexpr uint32_t mxfp4_block_size     = gptoss_mxfp4_block_size;
static constexpr uint32_t moe_output_tile_size = warp_size;
static constexpr uint32_t query_size           = gptoss_query_size;
static constexpr uint32_t kv_size              = gptoss_key_value_size;

static constexpr float attention_max_offset = 3.0f * 0.6931f;

static constexpr uint32_t mxfp4_padded_dim              = gptoss_mxfp4_padded_size;
static constexpr uint32_t mxfp4_value_row_bytes         = mxfp4_padded_dim / 2u;
static constexpr uint32_t mxfp4_scale_tail_offset       = intermediate_size / 2u;
static constexpr uint64_t qkv_values_bytes              = gptoss_qkv_values_size;
static constexpr uint64_t attention_output_values_bytes = gptoss_attention_output_values_size;

struct q8_0_row {
    const int8_t * values;
    const half *   scales;
};

struct mxfp4_row {
    const uint8_t * values;
    const uint8_t * scales;
};

struct moe_down_slot {
    mxfp4_row     row;
    const half *  activation;
    const float * bias;
    float         weight;
};

static __device__ __forceinline__ half * normalized_activation(const gptoss_decode_layer_params & p) {
    return p.activation_scratch;
}

static __device__ __forceinline__ half * expert_activations(const gptoss_decode_layer_params & p) {
    return p.activation_scratch + hidden_size;
}

static __device__ __forceinline__ float round_f16(float value) {
    return __half2float(__float2half_rn(value));
}

template <int width = warp_size> static __device__ __forceinline__ float warp_sum(float x) {
    static_assert(width == warp_size || width == warps_per_block, "unsupported reduction width");
#if defined(__gfx1201__)
    constexpr int  row_mask   = 0xf;
    constexpr int  bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;
    if constexpr (width == warp_size) {
        x += __shfl_xor(x, 16, warp_size);
        x += __builtin_bit_cast(
            float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x168, row_mask, bank_mask, bound_ctrl));
    }
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x164, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x162, row_mask, bank_mask, bound_ctrl));
    x += __builtin_bit_cast(
        float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x161, row_mask, bank_mask, bound_ctrl));
#else
#    pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, warp_size);
    }
#endif
    return x;
}

static __device__ __forceinline__ float block_sum(float value, float * warp_sums) {
    // All block threads participate; callers consume the result from warp 0, lane 0.
    value = warp_sum(value);
    if (threadIdx.x == 0) {
        warp_sums[threadIdx.y] = value;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        value = threadIdx.x < warps_per_block ? warp_sums[threadIdx.x] : 0.0f;
        value = warp_sum<warps_per_block>(value);
    }
    return value;
}

static __device__ __forceinline__ float warp_max(float x) {
#if defined(__gfx1201__)
    constexpr int  row_mask   = 0xf;
    constexpr int  bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;
    x                         = fmaxf(x, __shfl_xor(x, 16, warp_size));
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
    for (int off = warp_size / 2; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_xor(x, off, warp_size));
    }
#endif
    return x;
}

static __device__ __forceinline__ float e8m0_scale(uint8_t x) {
    const uint32_t bits = max((uint32_t) x << 23, 0x00400000u);
    return __uint_as_float(bits);
}

static __device__ __forceinline__ q8_0_row
q8_0_row_at(const int8_t * values, const half * scales, uint32_t row, uint32_t columns, uint32_t blocks) {
    return {
        values + (uint64_t) row * columns,
        scales + (uint64_t) row * blocks,
    };
}

static __device__ __forceinline__ q8_0_row q8_0_row_offset(const q8_0_row & row,
                                                           uint32_t         offset,
                                                           uint32_t         columns,
                                                           uint32_t         blocks) {
    return {
        row.values + (uint64_t) offset * columns,
        row.scales + (uint64_t) offset * blocks,
    };
}

static __device__ __forceinline__ mxfp4_row mxfp4_row_at(const uint8_t * values, uint64_t row) {
    const uint8_t * row_values = values + row * mxfp4_value_row_bytes;
    return {
        row_values,
        row_values + mxfp4_scale_tail_offset,
    };
}

static __device__ __forceinline__ mxfp4_row mxfp4_row_offset(const mxfp4_row & row, uint64_t offset) {
    return {
        row.values + offset * mxfp4_value_row_bytes,
        row.scales + offset * mxfp4_value_row_bytes,
    };
}

static __device__ __forceinline__ uint32_t load_u32_unaligned(const void * data, int word) {
    const uint8_t * bytes = (const uint8_t *) data;
    uint32_t        value;
    __builtin_memcpy(&value, bytes + 4 * word, sizeof(value));
    return value;
}

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ void mad_f16x2(float & acc, half2 v, half2 u) {
    acc = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, v), __builtin_bit_cast(f16x2, u), acc, false);
}

static __device__ __forceinline__ uint2 q8_to_f16x4(uint32_t packed) {
    // Convert four signed Q8 bytes to four FP16 values with packed permutes.
    packed ^= 0x80808080u;

    constexpr uint32_t exponent = 0x64646464u;
    const half2        offset   = __builtin_bit_cast(half2, 0xe480e480u);
    const half2        low  = __builtin_bit_cast(half2, __builtin_amdgcn_perm(packed, exponent, 0x00050004u)) + offset;
    const half2        high = __builtin_bit_cast(half2, __builtin_amdgcn_perm(packed, exponent, 0x00070006u)) + offset;

    return make_uint2(__builtin_bit_cast(uint32_t, low), __builtin_bit_cast(uint32_t, high));
}

static __device__ __forceinline__ float4 q8_0_dot4_segment(const q8_0_row & row0,
                                                           const q8_0_row & row1,
                                                           const q8_0_row & row2,
                                                           const q8_0_row & row3,
                                                           const half * __restrict__ x,
                                                           uint32_t block,
                                                           uint32_t segment) {
    constexpr uint32_t segment_size     = 8u;
    const uint32_t     element_offset   = segment_size * segment;
    const int8_t *     w0               = row0.values + (uint64_t) block * q8_block_size + element_offset;
    const int8_t *     w1               = row1.values + (uint64_t) block * q8_block_size + element_offset;
    const int8_t *     w2               = row2.values + (uint64_t) block * q8_block_size + element_offset;
    const int8_t *     w3               = row3.values + (uint64_t) block * q8_block_size + element_offset;
    const uint32_t *   activation_pairs = (const uint32_t *) (x + (uint64_t) block * q8_block_size + element_offset);
    float              acc0             = 0.0f;
    float              acc1             = 0.0f;
    float              acc2             = 0.0f;
    float              acc3             = 0.0f;
#pragma unroll
    for (int word = 0; word < 2; ++word) {
        const uint32_t q0 = load_u32_unaligned(w0, word);
        const uint32_t q1 = load_u32_unaligned(w1, word);
        const uint32_t q2 = load_u32_unaligned(w2, word);
        const uint32_t q3 = load_u32_unaligned(w3, word);
        const uint2    v0 = q8_to_f16x4(q0);
        const uint2    v1 = q8_to_f16x4(q1);
        const uint2    v2 = q8_to_f16x4(q2);
        const uint2    v3 = q8_to_f16x4(q3);
        const half2    x0 = __builtin_bit_cast(half2, activation_pairs[2 * word]);
        const half2    x1 = __builtin_bit_cast(half2, activation_pairs[2 * word + 1]);
        mad_f16x2(acc0, __builtin_bit_cast(half2, v0.x), x0);
        mad_f16x2(acc0, __builtin_bit_cast(half2, v0.y), x1);
        mad_f16x2(acc1, __builtin_bit_cast(half2, v1.x), x0);
        mad_f16x2(acc1, __builtin_bit_cast(half2, v1.y), x1);
        mad_f16x2(acc2, __builtin_bit_cast(half2, v2.x), x0);
        mad_f16x2(acc2, __builtin_bit_cast(half2, v2.y), x1);
        mad_f16x2(acc3, __builtin_bit_cast(half2, v3.x), x0);
        mad_f16x2(acc3, __builtin_bit_cast(half2, v3.y), x1);
    }

    return make_float4(__half2float(row0.scales[block]) * acc0, __half2float(row1.scales[block]) * acc1,
                       __half2float(row2.scales[block]) * acc2, __half2float(row3.scales[block]) * acc3);
}

static __device__ __forceinline__ float4 reduce_dot4(float acc0, float acc1, float acc2, float acc3) {
    constexpr int    partial_stride = (warps_per_block - 1) * warp_size;
    __shared__ float warp_partials[4 * partial_stride];
    const int        lane = threadIdx.x;
    const int        warp = threadIdx.y;

    if (warp > 0) {
        const int partial                           = (warp - 1) * warp_size + lane;
        warp_partials[0 * partial_stride + partial] = acc0;
        warp_partials[1 * partial_stride + partial] = acc1;
        warp_partials[2 * partial_stride + partial] = acc2;
        warp_partials[3 * partial_stride + partial] = acc3;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < warps_per_block - 1; ++w) {
            const int partial = w * warp_size + lane;
            acc0 += warp_partials[0 * partial_stride + partial];
            acc1 += warp_partials[1 * partial_stride + partial];
            acc2 += warp_partials[2 * partial_stride + partial];
            acc3 += warp_partials[3 * partial_stride + partial];
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
        acc2 = warp_sum(acc2);
        acc3 = warp_sum(acc3);
    }
    // Do not let the next Q8 task overwrite partials before warp 0 finishes.
    __syncthreads();
    return make_float4(acc0, acc1, acc2, acc3);
}

static __device__ float4 q8_0_dot4(const q8_0_row & row0,
                                   const q8_0_row & row1,
                                   const q8_0_row & row2,
                                   const q8_0_row & row3,
                                   const half * __restrict__ x,
                                   uint32_t blocks) {
    constexpr uint32_t segments_per_block = 4u;
    constexpr uint32_t block_stride       = block_size / segments_per_block;
    const uint32_t     tid                = threadIdx.y * warp_size + threadIdx.x;
    const uint32_t     segment            = tid % segments_per_block;
    float              acc0               = 0.0f;
    float              acc1               = 0.0f;
    float              acc2               = 0.0f;
    float              acc3               = 0.0f;
    for (uint32_t block = tid / segments_per_block; block < blocks; block += block_stride) {
        const float4 dot = q8_0_dot4_segment(row0, row1, row2, row3, x, block, segment);
        acc0 += dot.x;
        acc1 += dot.y;
        acc2 += dot.z;
        acc3 += dot.w;
    }

    return reduce_dot4(acc0, acc1, acc2, acc3);
}

static __device__ __forceinline__ uint4 mxfp4_to_f16x8(uint32_t codes) {
    // Decode four low and four high nibbles, then restore their interleaved order.
    constexpr uint32_t value_lut0    = 0x3e3c3800u;
    constexpr uint32_t value_lut1    = 0x46444240u;
    const uint32_t     low_magnitude = codes & 0x07070707u;
    const uint32_t     low_sign      = (codes & 0x08080808u) << 4;
    const uint32_t     low_bytes     = __builtin_amdgcn_perm(value_lut1, value_lut0, low_magnitude) | low_sign;
    const uint2        low_values    = make_uint2(__builtin_amdgcn_perm(low_bytes, 0u, 0x05010400u),
                                                  __builtin_amdgcn_perm(low_bytes, 0u, 0x07030602u));

    const uint32_t high_codes     = codes >> 4;
    const uint32_t high_magnitude = high_codes & 0x07070707u;
    const uint32_t high_sign      = (high_codes & 0x08080808u) << 4;
    const uint32_t high_bytes     = __builtin_amdgcn_perm(value_lut1, value_lut0, high_magnitude) | high_sign;
    const uint2    high_values    = make_uint2(__builtin_amdgcn_perm(high_bytes, 0u, 0x05010400u),
                                               __builtin_amdgcn_perm(high_bytes, 0u, 0x07030602u));

    return make_uint4((low_values.x & 0x0000ffffu) | (high_values.x << 16),
                      (low_values.x >> 16) | (high_values.x & 0xffff0000u),
                      (low_values.y & 0x0000ffffu) | (high_values.y << 16),
                      (low_values.y >> 16) | (high_values.y & 0xffff0000u));
}

static __device__ __forceinline__ float mxfp4_dot_segment(const mxfp4_row & row,
                                                          const half * __restrict__ x,
                                                          uint32_t block,
                                                          uint32_t segment) {
    const uint8_t *  values           = row.values + (uint64_t) block * (mxfp4_block_size / 2u);
    const uint2      packed           = *(const uint2 *) (values + 8u * segment);
    const uint4      values0          = mxfp4_to_f16x8(packed.x);
    const uint4      values1          = mxfp4_to_f16x8(packed.y);
    const uint32_t * activation_pairs = (const uint32_t *) (x + (uint64_t) block * mxfp4_block_size + 16u * segment);
    float            sum              = 0.0f;
    mad_f16x2(sum, __builtin_bit_cast(half2, values0.x), __builtin_bit_cast(half2, activation_pairs[0]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values0.y), __builtin_bit_cast(half2, activation_pairs[1]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values0.z), __builtin_bit_cast(half2, activation_pairs[2]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values0.w), __builtin_bit_cast(half2, activation_pairs[3]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values1.x), __builtin_bit_cast(half2, activation_pairs[4]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values1.y), __builtin_bit_cast(half2, activation_pairs[5]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values1.z), __builtin_bit_cast(half2, activation_pairs[6]));
    mad_f16x2(sum, __builtin_bit_cast(half2, values1.w), __builtin_bit_cast(half2, activation_pairs[7]));
    return e8m0_scale(row.scales[block]) * sum;
}

static __device__ float2 mxfp4_dot2(const mxfp4_row & row0, const mxfp4_row & row1, const half * __restrict__ x) {
    constexpr uint32_t blocks             = intermediate_size / mxfp4_block_size;
    constexpr uint32_t segments_per_block = 2u;
    constexpr uint32_t block_stride       = warp_size / segments_per_block;
    const uint32_t     lane               = threadIdx.x;
    const uint32_t     segment            = lane % segments_per_block;
    float              acc0               = 0.0f;
    float              acc1               = 0.0f;
    for (uint32_t block = lane / segments_per_block; block < blocks; block += block_stride) {
        acc0 += mxfp4_dot_segment(row0, x, block, segment);
        acc1 += mxfp4_dot_segment(row1, x, block, segment);
    }
    return make_float2(warp_sum(acc0), warp_sum(acc1));
}

static __device__ float4 mxfp4_dot4(const mxfp4_row & row0,
                                    const mxfp4_row & row1,
                                    const mxfp4_row & row2,
                                    const mxfp4_row & row3,
                                    const half * __restrict__ x) {
    constexpr uint32_t blocks             = hidden_size / mxfp4_block_size;
    constexpr uint32_t segments_per_block = 2u;
    constexpr uint32_t block_stride       = warp_size / segments_per_block;
    const uint32_t     lane               = threadIdx.x;
    const uint32_t     segment            = lane % segments_per_block;
    float              acc0               = 0.0f;
    float              acc1               = 0.0f;
    float              acc2               = 0.0f;
    float              acc3               = 0.0f;
    for (uint32_t block = lane / segments_per_block; block < blocks; block += block_stride) {
        acc0 += mxfp4_dot_segment(row0, x, block, segment);
        acc1 += mxfp4_dot_segment(row1, x, block, segment);
        acc2 += mxfp4_dot_segment(row2, x, block, segment);
        acc3 += mxfp4_dot_segment(row3, x, block, segment);
    }
    return make_float4(warp_sum(acc0), warp_sum(acc1), warp_sum(acc2), warp_sum(acc3));
}

static __device__ __forceinline__ float swiglu_oai(float gate, float up) {
    gate            = fminf(gate, 7.0f);
    up              = fmaxf(fminf(up, 7.0f), -7.0f);
    const float e   = expf(-1.702f * gate);
    const float glu = gate / (1.0f + e);
    return glu * (1.0f + up);
}

static __device__ __forceinline__ float add_no_contract(float a, float b) {
    _Pragma("clang fp contract(off)");
    return a + b;
}

static __device__ __forceinline__ float mul_no_contract(float a, float b) {
    _Pragma("clang fp contract(off)");
    return a * b;
}

static __device__ void attention_rms_norm(const gptoss_decode_layer_params & p) {
    const uint32_t tid         = threadIdx.y * warp_size + threadIdx.x;
    const uint32_t logical_tid = blockIdx.x * block_size + tid;

    __shared__ float rms_scale;
    if (!p.reuse_attention_rms) {
        if (blockIdx.x != 0) {
            return;
        }

        __shared__ float warp_partials[warps_per_block];
        float            sum = 0.0f;
        for (uint32_t i = tid; i < hidden_size; i += block_size) {
            const float value = p.cur[i];
            sum += value * value;
        }
        sum = block_sum(sum, warp_partials);
        if (tid == 0) {
            rms_scale = rsqrtf(sum / (float) hidden_size + p.rms_epsilon);
        }
        __syncthreads();

        for (uint32_t i = tid; i < hidden_size; i += block_size) {
            normalized_activation(p)[i] = __float2half_rn(p.cur[i] * rms_scale * p.attn_norm[i]);
        }
        return;
    }

    if (threadIdx.y == 0) {
        float total = 0.0f;
        for (uint32_t i = threadIdx.x; i < gridDim.x; i += warp_size) {
            total += p.rms_partials[i];
        }
        total = warp_sum(total);
        if (threadIdx.x == 0) {
            rms_scale = rsqrtf(total / (float) hidden_size + p.rms_epsilon);
        }
    }
    __syncthreads();

    const float    scale        = rms_scale;
    const uint32_t thread_count = gridDim.x * block_size;
    for (uint32_t i = logical_tid; i < hidden_size; i += thread_count) {
        normalized_activation(p)[i] = __float2half_rn(p.cur[i] * scale * p.attn_norm[i]);
    }
}

static __device__ __forceinline__ float rope_ramp(float low, float high, uint32_t pair) {
    const float y = ((float) pair - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ __forceinline__ void rope_pair(const gptoss_decode_layer_params & p,
                                                 uint32_t                           pair,
                                                 float                              x0,
                                                 float                              x1,
                                                 float &                            y0,
                                                 float &                            y1) {
    const float theta_extrap = (float) p.position * powf(p.rope_theta_scale, (float) pair);
    const float theta_interp = p.rope_freq_scale * theta_extrap;
    float       theta        = theta_interp;
    float       magnitude    = p.rope_attn_factor;

    if (p.rope_ext_factor != 0.0f) {
        const float mix = rope_ramp(p.rope_corr_low, p.rope_corr_high, pair) * p.rope_ext_factor;
        theta           = theta_interp * (1.0f - mix) + theta_extrap * mix;
        magnitude *= 1.0f + 0.1f * logf(1.0f / p.rope_freq_scale);
    }

    const float cosine = cosf(theta) * magnitude;
    const float sine   = sinf(theta) * magnitude;
    // Round the projected Q/K pair to FP16 before the FP32 RoPE rotation.
    x0                 = round_f16(x0);
    x1                 = round_f16(x1);
    y0                 = fmaf(x0, cosine, -(x1 * sine));
    y1                 = fmaf(x1, cosine, x0 * sine);
}

// Project Q/K/V from the normalized FP16 row, apply RoPE, and update the FP16
// query scratch and KV cache.
static __device__ void qkv_rope_cache(const gptoss_decode_layer_params & p) {
    constexpr uint32_t pairs_per_head      = head_size / 2u;
    constexpr uint32_t pair_quads_per_head = pairs_per_head / 2u;
    constexpr uint32_t q_pair_quads        = query_head_count * pair_quads_per_head;
    constexpr uint32_t k_pair_quads        = kv_head_count * pair_quads_per_head;
    constexpr uint32_t qk_tasks            = q_pair_quads + k_pair_quads;
    constexpr uint32_t v_quads             = kv_size / 4u;
    constexpr uint32_t total_tasks         = qk_tasks + v_quads;
    const uint32_t     blocks_per_row      = hidden_size / q8_block_size;
    const half * __restrict__ qkv_scales   = (const half *) (p.qkv_values + qkv_values_bytes);
    const uint64_t kv_base                 = (uint64_t) p.kv_write_row * kv_size;
    half * __restrict__ scaled_query       = p.query;

    for (uint32_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        if (task < qk_tasks) {
            const bool     is_q             = task < q_pair_quads;
            const uint32_t pair_quad        = is_q ? task : task - q_pair_quads;
            const uint32_t head             = pair_quad / pair_quads_per_head;
            const uint32_t pair0            = (pair_quad % pair_quads_per_head) * 2u;
            const uint32_t row0             = head * head_size + pair0;
            const uint32_t row_base         = is_q ? 0u : query_size;
            const float * __restrict__ bias = is_q ? p.attn_q_bias : p.attn_k_bias;

            const q8_0_row weight0 =
                q8_0_row_at(p.qkv_values, qkv_scales, row_base + row0, hidden_size, blocks_per_row);
            const q8_0_row weight1 = q8_0_row_offset(weight0, pairs_per_head, hidden_size, blocks_per_row);
            const q8_0_row weight2 = q8_0_row_offset(weight0, 1u, hidden_size, blocks_per_row);
            const q8_0_row weight3 = q8_0_row_offset(weight1, 1u, hidden_size, blocks_per_row);

            const float4 dot = q8_0_dot4(weight0, weight1, weight2, weight3, normalized_activation(p), blocks_per_row);

            if (threadIdx.y == 0 && threadIdx.x < 2) {
                const uint32_t pair_in_quad = threadIdx.x;
                const uint32_t pair         = pair0 + pair_in_quad;
                const uint32_t low          = row0 + pair_in_quad;
                const uint32_t high         = low + pairs_per_head;
                const float    x0           = (pair_in_quad == 0u ? dot.x : dot.z) + bias[low];
                const float    x1           = (pair_in_quad == 0u ? dot.y : dot.w) + bias[high];
                float          y0;
                float          y1;
                rope_pair(p, pair, x0, x1, y0, y1);
                if (is_q) {
                    scaled_query[low]  = __float2half_rn(y0 * 0.125f);
                    scaled_query[high] = __float2half_rn(y1 * 0.125f);
                } else {
                    p.cache_k[kv_base + low]  = __float2half_rn(y0);
                    p.cache_k[kv_base + high] = __float2half_rn(y1);
                }
            }
        } else {
            const uint32_t v_quad     = task - qk_tasks;
            const uint32_t row        = v_quad * 4u;
            const uint32_t weight_row = query_size + kv_size + row;
            const q8_0_row weight0    = q8_0_row_at(p.qkv_values, qkv_scales, weight_row, hidden_size, blocks_per_row);
            const q8_0_row weight1    = q8_0_row_offset(weight0, 1u, hidden_size, blocks_per_row);
            const q8_0_row weight2    = q8_0_row_offset(weight1, 1u, hidden_size, blocks_per_row);
            const q8_0_row weight3    = q8_0_row_offset(weight2, 1u, hidden_size, blocks_per_row);
            const float4 dot = q8_0_dot4(weight0, weight1, weight2, weight3, normalized_activation(p), blocks_per_row);
            if (threadIdx.y == 0 && threadIdx.x == 0) {
                p.cache_v[kv_base + row]      = __float2half_rn(add_no_contract(dot.x, p.attn_v_bias[row]));
                p.cache_v[kv_base + row + 1u] = __float2half_rn(add_no_contract(dot.y, p.attn_v_bias[row + 1u]));
                p.cache_v[kv_base + row + 2u] = __float2half_rn(add_no_contract(dot.z, p.attn_v_bias[row + 2u]));
                p.cache_v[kv_base + row + 3u] = __float2half_rn(add_no_contract(dot.w, p.attn_v_bias[row + 3u]));
            }
        }
    }
}

// The active window can intersect at most two globally aligned flash tiles.
// Keep both partials within one Q-head block, then combine them in the same
// partition order as standalone flash attention. Before the window fills it
// naturally covers the complete history.
static __device__ void sliding_window_attention(const gptoss_decode_layer_params & p) {
    const uint32_t lane       = threadIdx.x;
    const uint32_t warp       = threadIdx.y;
    const uint32_t query_head = blockIdx.x;
    if (query_head >= query_head_count) {
        return;
    }

    constexpr uint32_t tile_keys = 128u;
    static_assert(sliding_window_size == tile_keys, "SWA window must cover one flash tile");
    static_assert(sliding_window_size <= block_size);
    __shared__ float  scores[2u * tile_keys];
    __shared__ float2 tile_meta[2];

    const uint32_t gqa              = query_head_count / kv_head_count;
    const uint32_t kv_head          = query_head / gqa;
    const half *   query            = p.query + (uint64_t) query_head * head_size;
    const uint32_t tid              = warp * warp_size + lane;
    const uint32_t history_count    = (uint32_t) p.position + 1u;
    const uint32_t window_count     = p.n_kv;
    const uint32_t window_start     = history_count - window_count;
    const uint32_t first_tile       = window_start / tile_keys;
    const uint32_t first_tile_begin = window_start % tile_keys;
    const uint32_t tile_span        = first_tile_begin + window_count;
    const uint32_t tile_count       = (tile_span + tile_keys - 1u) / tile_keys;
    // The host limits n_kv to sliding_window_size, so each thread owns at most one key.
    if (tid < window_count) {
        const uint32_t score_index = first_tile_begin + tid;
        const uint32_t row         = (uint32_t) p.kv_rows[tid];
        const half *   key         = p.cache_k + (uint64_t) row * kv_size + (uint64_t) kv_head * head_size;
        float          dot         = 0.0f;
#pragma unroll
        for (uint32_t d = 0; d < head_size; d += 2u) {
            mad_f16x2(dot, __halves2half2(key[d], key[d + 1u]), __halves2half2(query[d], query[d + 1u]));
        }
        scores[score_index] = dot;
    }
    __syncthreads();

    if (warp < tile_count) {
        const uint32_t begin      = warp == 0u ? first_tile_begin : 0u;
        const uint32_t tile_end   = tile_span - warp * tile_keys;
        const uint32_t end        = tile_end < tile_keys ? tile_end : tile_keys;
        const uint32_t score_base = warp * tile_keys;

        float kq_max = -FLT_MAX / 2.0f;
        for (uint32_t key_in_tile = lane; key_in_tile < tile_keys; key_in_tile += warp_size) {
            if (key_in_tile >= begin && key_in_tile < end) {
                kq_max = fmaxf(kq_max, scores[score_base + key_in_tile] + attention_max_offset);
            }
        }
        kq_max = warp_max(kq_max);

        float kq_sum = 0.0f;
        for (uint32_t key_in_tile = lane; key_in_tile < tile_keys; key_in_tile += warp_size) {
            if (key_in_tile >= begin && key_in_tile < end) {
                const float w                    = expf(scores[score_base + key_in_tile] - kq_max);
                scores[score_base + key_in_tile] = w;
                kq_sum += w;
            }
        }
        kq_sum = warp_sum(kq_sum);

        half2 value_acc = make_half2(0.0f, 0.0f);
#pragma unroll 8
        for (uint32_t key_in_tile = begin; key_in_tile < end; ++key_in_tile) {
            const uint32_t window_index = score_base + key_in_tile - first_tile_begin;
            const uint32_t row          = (uint32_t) p.kv_rows[window_index];
            const half *   value        = p.cache_v + (uint64_t) row * kv_size + (uint64_t) kv_head * head_size;
            const float    w            = scores[score_base + key_in_tile];
            value_acc += __halves2half2(value[lane], value[lane + warp_size]) * make_half2(w, w);
        }

        const uint32_t part = (first_tile + warp) % p.attn_parallel_blocks;
        if (part == 0u) {
            const float sink       = p.attn_sinks[query_head];
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

    if (warp == 0) {
        const uint32_t part0         = first_tile % p.attn_parallel_blocks;
        const uint32_t part1         = (first_tile + 1u) % p.attn_parallel_blocks;
        const bool     sink_has_tile = part0 == 0u || (tile_count == 2u && part1 == 0u);
        float kq_max = sink_has_tile ? (part0 == 0u ? tile_meta[0].x : tile_meta[1].x) : p.attn_sinks[query_head];
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
                meta_max = p.attn_sinks[query_head];
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
        const float out0         = numerator0 / denominator;
        const float out1         = numerator1 / denominator;
        half *      output       = p.query + (uint64_t) query_head * head_size;
        output[lane]             = __float2half_rn(out0);
        output[lane + warp_size] = __float2half_rn(out1);
    }
}

// Full causal decode attention inside the cooperative layer launch. The task
// layout matches flash_attn_tile: one task owns a KV head and one interleaved
// 128-key partition, while its eight warps own the eight GQA query heads. K and
// V are streamed through 32-key chunks so the layer kernel does not inherit the
// standalone flash kernel's 20+ KiB shared tile.
static __device__ __forceinline__ uint4 load_cache_segment(const half *                       cache,
                                                           const gptoss_decode_layer_params & p,
                                                           uint32_t                           key,
                                                           uint32_t                           kv_head,
                                                           uint32_t                           segment) {
    if (key >= p.n_kv) {
        return make_uint4(0u, 0u, 0u, 0u);
    }

    const uint32_t row = (uint32_t) p.kv_rows[key];
    return *(const uint4 *) (cache + (uint64_t) row * kv_size + (uint64_t) kv_head * head_size + 8u * segment);
}

static __device__ __forceinline__ void store_shared_segment(half *   shared,
                                                            uint32_t stride,
                                                            uint32_t key,
                                                            uint32_t segment,
                                                            uint4    value) {
    uint32_t * destination = (uint32_t *) (shared + key * stride + 8u * segment);
    destination[0]         = value.x;
    destination[1]         = value.y;
    destination[2]         = value.z;
    destination[3]         = value.w;
}

static __device__ void full_attention_parts(const gptoss_decode_layer_params & p) {
    constexpr uint32_t tile_keys        = 128u;
    constexpr uint32_t chunk_keys       = 32u;
    // A 64-half row is exactly 128 bytes and maps the same dimension of all
    // 32 keys to one LDS bank. Two pad halves make the half2 row stride 33
    // banks, rotating successive keys across every bank.
    constexpr uint32_t kv_shared_stride = head_size + 2u;
    constexpr uint32_t gqa              = query_head_count / kv_head_count;
    static_assert(gqa == warps_per_block, "one warp must own each GQA query head");
    static_assert(head_size == 64u, "streamed full attention is specialized to D=64");
    static_assert(chunk_keys * (head_size / 8u) == block_size, "one thread must copy one 16-byte KV segment");

    __shared__ half kv_chunk[2u * chunk_keys * kv_shared_stride];
    __shared__ half weights[warps_per_block * tile_keys];

    const half * __restrict__ queries = p.query;
    const half * __restrict__ cache_k = p.cache_k;
    const half * __restrict__ cache_v = p.cache_v;
    const uint32_t parallel_blocks    = p.attn_parallel_blocks;
    const uint32_t active_count       = p.n_kv;
    const uint32_t lane               = threadIdx.x;
    const uint32_t warp               = threadIdx.y;
    const uint32_t tid                = warp * warp_size + lane;
    const uint32_t task_count         = kv_head_count * parallel_blocks;
    const uint32_t tile_stride        = parallel_blocks * tile_keys;

    for (uint32_t task = blockIdx.x; task < task_count; task += gridDim.x) {
        const uint32_t kv_head      = task / parallel_blocks;
        const uint32_t part         = task - kv_head * parallel_blocks;
        const uint32_t query_head   = kv_head * gqa + warp;
        const half *   query        = queries + (uint64_t) query_head * head_size;
        half2 *        shared_query = (half2 *) (weights + warp * tile_keys);

        float kq_max    = -FLT_MAX / 2.0f;
        float kq_sum    = 0.0f;
        half2 value_acc = make_half2(0.0f, 0.0f);

        for (uint32_t tile = part * tile_keys; tile < active_count; tile += tile_stride) {
            /* Softmax weights reuse this LDS row, so restore Q before every
               interleaved tile owned by the partition. */
            shared_query[lane] = *(const half2 *) (query + 2u * lane);

            float          scores[tile_keys / warp_size];
            const uint32_t valid_chunks = (min(tile_keys, active_count - tile) + chunk_keys - 1u) / chunk_keys;

            const uint32_t copy_key     = tid / (head_size / 8u);
            const uint32_t copy_segment = tid - copy_key * (head_size / 8u);
            const uint32_t first_key    = tile + copy_key;
            const uint4    first_k      = load_cache_segment(cache_k, p, first_key, kv_head, copy_segment);
            store_shared_segment(kv_chunk, kv_shared_stride, copy_key, copy_segment, first_k);
            __syncthreads();

#pragma unroll 1
            for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
                const bool has_next = chunk + 1u < valid_chunks;
                uint4      next_k   = make_uint4(0u, 0u, 0u, 0u);
                if (has_next) {
                    const uint32_t next_key = tile + (chunk + 1u) * chunk_keys + copy_key;
                    next_k                  = load_cache_segment(cache_k, p, next_key, kv_head, copy_segment);
                }

                const half * current_k = kv_chunk + (chunk % 2u) * chunk_keys * kv_shared_stride;

                float dot = 0.0f;
#pragma unroll
                for (uint32_t d = 0; d < head_size; d += 8u) {
                    const half2 q0  = shared_query[d / 2u];
                    const half2 q1  = shared_query[d / 2u + 1u];
                    const half2 q2  = shared_query[d / 2u + 2u];
                    const half2 q3  = shared_query[d / 2u + 3u];
                    const half2 k00 = *(const half2 *) (current_k + lane * kv_shared_stride + d);
                    const half2 k01 = *(const half2 *) (current_k + lane * kv_shared_stride + d + 2u);
                    const half2 k02 = *(const half2 *) (current_k + lane * kv_shared_stride + d + 4u);
                    const half2 k03 = *(const half2 *) (current_k + lane * kv_shared_stride + d + 6u);
                    mad_f16x2(dot, k00, q0);
                    mad_f16x2(dot, k01, q1);
                    mad_f16x2(dot, k02, q2);
                    mad_f16x2(dot, k03, q3);
                }
                scores[chunk] = dot;

                if (has_next) {
                    half * next_k_shared = kv_chunk + ((chunk + 1u) % 2u) * chunk_keys * kv_shared_stride;
                    store_shared_segment(next_k_shared, kv_shared_stride, copy_key, copy_segment, next_k);
                }
                __syncthreads();
            }

            float kq_max_new = kq_max;
#pragma unroll
            for (uint32_t i = 0; i < tile_keys / warp_size; ++i) {
                const uint32_t key = tile + i * warp_size + lane;
                if (key < active_count) {
                    kq_max_new = fmaxf(kq_max_new, scores[i] + attention_max_offset);
                }
            }
            kq_max_new = warp_max(kq_max_new);

            const float kq_max_scale = expf(kq_max - kq_max_new);
            kq_max                   = kq_max_new;
            float kq_sum_add         = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < tile_keys / warp_size; ++i) {
                const uint32_t key_in_tile = i * warp_size + lane;
                const uint32_t key         = tile + key_in_tile;
                const float    w           = key < active_count ? expf(scores[i] - kq_max) : 0.0f;
                kq_sum_add += w;
                weights[warp * tile_keys + key_in_tile] = (half) w;
            }
            kq_sum                  = kq_sum * kq_max_scale + kq_sum_add;
            const half2 value_scale = make_half2(kq_max_scale, kq_max_scale);

            const uint4 first_v = load_cache_segment(cache_v, p, first_key, kv_head, copy_segment);
            store_shared_segment(kv_chunk, kv_shared_stride, copy_key, copy_segment, first_v);
            __syncthreads();

#pragma unroll 1
            for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
                const bool has_next = chunk + 1u < valid_chunks;
                uint4      next_v   = make_uint4(0u, 0u, 0u, 0u);
                if (has_next) {
                    const uint32_t next_key = tile + (chunk + 1u) * chunk_keys + copy_key;
                    next_v                  = load_cache_segment(cache_v, p, next_key, kv_head, copy_segment);
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
                    half * next_v_shared = kv_chunk + ((chunk + 1u) % 2u) * chunk_keys * kv_shared_stride;
                    store_shared_segment(next_v_shared, kv_shared_stride, copy_key, copy_segment, next_v);
                }
                __syncthreads();
            }
        }

        kq_sum = warp_sum(kq_sum);
        if (part == 0u) {
            const float sink         = p.attn_sinks[query_head];
            const float kq_max_new   = fmaxf(kq_max, sink);
            const float kq_max_scale = expf(kq_max - kq_max_new);
            kq_max                   = kq_max_new;
            kq_sum                   = kq_sum * kq_max_scale + expf(sink - kq_max);
            value_acc *= make_half2(kq_max_scale, kq_max_scale);
        }

        const uint64_t part_index                = (uint64_t) query_head * parallel_blocks + part;
        const uint64_t part_base                 = part_index * head_size;
        const float2   value                     = __half22float2(value_acc);
        p.attn_parts[part_base + 2u * lane]      = value.x;
        p.attn_parts[part_base + 2u * lane + 1u] = value.y;
        if (lane == 0u) {
            p.attn_meta[part_index] = make_float2(kq_max, kq_sum);
        }
    }
}

static __device__ void attention_output_residual_rms(const gptoss_decode_layer_params & p) {
    const uint32_t blocks_per_row    = query_size / q8_block_size;
    const uint32_t total_quads       = hidden_size >> 2;
    const half * __restrict__ scales = (const half *) (p.attn_output_values + attention_output_values_bytes);
    float ffn_rms_partial            = 0.0f;
    for (uint32_t quad = blockIdx.x; quad < total_quads; quad += gridDim.x) {
        const uint32_t row  = quad << 2;
        const q8_0_row row0 = q8_0_row_at(p.attn_output_values, scales, row, query_size, blocks_per_row);
        const q8_0_row row1 = q8_0_row_offset(row0, 1u, query_size, blocks_per_row);
        const q8_0_row row2 = q8_0_row_offset(row1, 1u, query_size, blocks_per_row);
        const q8_0_row row3 = q8_0_row_offset(row2, 1u, query_size, blocks_per_row);
        const float4   dot  = q8_0_dot4(row0, row1, row2, row3, p.query, blocks_per_row);
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
        p.rms_partials[blockIdx.x] = ffn_rms_partial;
    }
}

static __device__ void full_attention_combine(const gptoss_decode_layer_params & p) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid  = threadIdx.y * warp_size + threadIdx.x;
    if (head >= query_head_count || tid >= head_size) {
        return;
    }

    const uint32_t parallel_blocks = p.attn_parallel_blocks;
    const float *  parts           = p.attn_parts + (uint64_t) head * parallel_blocks * head_size;
    const float2 * meta            = p.attn_meta + (uint64_t) head * parallel_blocks;

    float kq_max = meta[0].x;
    for (uint32_t part = 1; part < parallel_blocks; ++part) {
        kq_max = fmaxf(kq_max, meta[part].x);
    }

    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (uint32_t part = 0; part < parallel_blocks; ++part) {
        const float scale = expf(meta[part].x - kq_max);
        numerator += scale * parts[(uint64_t) part * head_size + tid];
        denominator += scale * meta[part].y;
    }
    const float output = numerator / denominator;

    p.query[(uint64_t) head * head_size + tid] = __float2half_rn(output);
}

static __device__ void post_attention_norm_and_router(const gptoss_decode_layer_params & p) {
    const uint32_t tid         = threadIdx.y * warp_size + threadIdx.x;
    const uint32_t logical_tid = blockIdx.x * block_size + tid;

    __shared__ float rms_scale;
    if (threadIdx.y == 0) {
        float total = 0.0f;
        for (uint32_t i = threadIdx.x; i < gridDim.x; i += warp_size) {
            total += p.rms_partials[i];
        }
        total = warp_sum(total);
        if (threadIdx.x == 0) {
            rms_scale = rsqrtf(total / (float) hidden_size + p.rms_epsilon);
        }
    }
    __syncthreads();

    const float    scale        = rms_scale;
    const uint32_t thread_count = gridDim.x * block_size;
    for (uint32_t i = logical_tid; i < hidden_size; i += thread_count) {
        normalized_activation(p)[i] = __float2half_rn(p.next[i] * scale * p.post_attention_norm[i]);
    }

    const uint32_t   column_pairs = hidden_size / 2u;
    __shared__ float router_warp_sums[warps_per_block];
    for (uint32_t expert = blockIdx.x; expert < expert_count; expert += gridDim.x) {
        // Router weights consume FP32 normalization; FP16 scratch feeds the MXFP4 projections.
        const float2 * weight_pairs = (const float2 *) (p.router_weight + expert * hidden_size);
        float          acc          = 0.0f;
        for (uint32_t column = tid; column < column_pairs; column += block_size) {
            const uint32_t i           = 2u * column;
            const float2   weight_pair = weight_pairs[column];
            const float    normalized0 = p.next[i] * scale * p.post_attention_norm[i];
            const float    normalized1 = p.next[i + 1u] * scale * p.post_attention_norm[i + 1u];
            acc += weight_pair.x * normalized0;
            acc += weight_pair.y * normalized1;
        }
        acc = block_sum(acc, router_warp_sums);
        if (tid == 0) {
            p.router[expert] = acc + p.router_bias[expert];
        }
        // Protect shared partials only when this block will process another expert.
        if (gridDim.x < expert_count) {
            __syncthreads();
        }
    }
}

static __device__ void select_experts(const gptoss_decode_layer_params & p, int32_t * __restrict__ ids) {
    if (threadIdx.y != 0) {
        return;
    }

    const uint32_t lane  = threadIdx.x;
    float          score = p.router[lane];
    if (isnan(score)) {
        score = -FLT_MAX;
    }

#pragma unroll
    for (uint32_t selected = 0; selected < experts_used; ++selected) {
        float    best_score  = score;
        uint32_t best_expert = lane;

#pragma unroll
        for (uint32_t offset = warp_size / 2; offset > 0; offset >>= 1) {
            const float    other_score  = __shfl_xor(best_score, offset, warp_size);
            const uint32_t other_expert = __shfl_xor(best_expert, offset, warp_size);
            if (other_score > best_score || (other_score == best_score && other_expert < best_expert)) {
                best_score  = other_score;
                best_expert = other_expert;
            }
        }

        if (lane == 0) {
            ids[selected] = (int32_t) best_expert;
        }
        if (lane == best_expert) {
            score = -INFINITY;
        }
    }

    if (blockIdx.x != 0) {
        return;
    }

    __syncwarp();
    const int32_t expert = lane < experts_used ? ids[lane] : 0;
    float         logit  = lane < experts_used ? p.router[expert] : -INFINITY;
    if (isnan(logit)) {
        logit = -FLT_MAX;
    }
    const float maximum = warp_max(logit);
    const float weight  = lane < experts_used ? expf(logit - maximum) : 0.0f;
    const float sum     = warp_sum(weight);

    if (lane < experts_used) {
        p.expert_ids[lane]     = expert;
        p.expert_weights[lane] = weight / sum;
    }
}

static __device__ void moe_gate_up(const gptoss_decode_layer_params & p) {
    const uint32_t out_blocks            = intermediate_size / moe_output_tile_size;
    const uint32_t total_blocks          = experts_used * out_blocks;
    const uint32_t lane                  = threadIdx.x;
    const half * __restrict__ normalized = normalized_activation(p);
    half * __restrict__ expert_output    = expert_activations(p);
    __shared__ int32_t ids[experts_used];
    __shared__ float   act[moe_output_tile_size];

    if (blockIdx.x >= total_blocks) {
        return;
    }

    select_experts(p, ids);
    __syncthreads();

    for (uint32_t task = blockIdx.x; task < total_blocks; task += gridDim.x) {
        const uint32_t slot         = task / out_blocks;
        const uint32_t output_block = task % out_blocks;

        const uint32_t  expert              = (uint32_t) ids[slot];
        const uint32_t  row_base            = output_block * moe_output_tile_size;
        const uint64_t  logical_expert_row  = (uint64_t) expert * intermediate_size + row_base;
        const uint64_t  physical_expert_row = (uint64_t) expert * 2u * mxfp4_padded_dim + 2u * row_base;
        const mxfp4_row gate_tile           = mxfp4_row_at(p.moe_gate_up_values, physical_expert_row);
        const mxfp4_row up_tile             = mxfp4_row_offset(gate_tile, 1u);
        const float * __restrict__ bias     = p.moe_gate_up_bias + 2u * logical_expert_row;

        for (uint32_t row_in_block = 2u * threadIdx.y; row_in_block < moe_output_tile_size;
             row_in_block += 2u * warps_per_block) {
            const mxfp4_row gate0 = mxfp4_row_offset(gate_tile, 2u * row_in_block);
            const mxfp4_row up0   = mxfp4_row_offset(up_tile, 2u * row_in_block);
            const mxfp4_row gate1 = mxfp4_row_offset(gate0, 2u);
            const mxfp4_row up1   = mxfp4_row_offset(up0, 2u);
            const float4    dots  = mxfp4_dot4(gate0, up0, gate1, up1, normalized);
            if (lane == 0) {
                const float gate       = dots.x + bias[2u * row_in_block];
                const float up         = dots.y + bias[2u * row_in_block + 1u];
                act[row_in_block]      = swiglu_oai(gate, up);
                const float gate_next  = dots.z + bias[2u * row_in_block + 2u];
                const float up_next    = dots.w + bias[2u * row_in_block + 3u];
                act[row_in_block + 1u] = swiglu_oai(gate_next, up_next);
            }
        }
        __syncthreads();

        if (threadIdx.y == 0) {
            expert_output[slot * intermediate_size + row_base + lane] = __float2half_rn(act[lane]);
        }
        if (task + gridDim.x < total_blocks) {
            __syncthreads();
        }
    }
}

static __device__ void moe_down_residual_rms(const gptoss_decode_layer_params & p) {
    const uint32_t warp_global            = blockIdx.x * warps_per_block + threadIdx.y;
    const uint32_t warp_count             = gridDim.x * warps_per_block;
    const uint32_t lane                   = threadIdx.x;
    const half * __restrict__ activations = expert_activations(p);
    float * __restrict__ next             = p.next;
    __shared__ float rms_warp_partials[warps_per_block];

    if (blockIdx.x * warps_per_block >= hidden_size) {
        return;
    }

    moe_down_slot slots[experts_used];
#pragma unroll
    for (uint32_t slot = 0; slot < experts_used; ++slot) {
        const uint32_t expert = (uint32_t) p.expert_ids[slot];
        slots[slot]           = {
            mxfp4_row_at(p.moe_down_values, (uint64_t) expert * mxfp4_padded_dim),
            activations + (uint64_t) slot * intermediate_size,
            p.moe_down_bias + (uint64_t) expert * hidden_size,
            p.expert_weights[slot],
        };
    }

    constexpr uint32_t row_pair_count = hidden_size / 2u;
    float              rms_partial    = 0.0f;
    for (uint32_t pair = warp_global; pair < row_pair_count; pair += warp_count) {
        const uint32_t row0 = 2u * pair;
        float2         dots[experts_used];
#pragma unroll
        for (uint32_t slot = 0; slot < experts_used; ++slot) {
            const mxfp4_row row = mxfp4_row_offset(slots[slot].row, row0);
            dots[slot]          = mxfp4_dot2(row, mxfp4_row_offset(row, 1u), slots[slot].activation);
        }

        if (lane < 2) {
            const uint32_t output_row  = row0 + lane;
            const float    first_value = lane == 0 ? dots[0].x : dots[0].y;
            float total = mul_no_contract(slots[0].weight, add_no_contract(first_value, slots[0].bias[output_row]));
#pragma unroll
            for (uint32_t slot = 1; slot < experts_used; ++slot) {
                const float value = lane == 0 ? dots[slot].x : dots[slot].y;
                total             = add_no_contract(
                    total, mul_no_contract(slots[slot].weight, add_no_contract(value, slots[slot].bias[output_row])));
            }

            const float result = add_no_contract(next[output_row], total);
            next[output_row]   = result;
            rms_partial += result * result;
        }
    }

    const float block_partial = block_sum(rms_partial, rms_warp_partials);
    if (threadIdx.y == 0 && lane == 0) {
        p.rms_partials[blockIdx.x] = block_partial;
    }
}

}  // namespace

// Complete sliding-window decode layer.
__launch_bounds__(block_size, 1) __global__ void gptoss_decode_layer_swa_kernel(gptoss_decode_layer_params p) {
    cg::grid_group grid = cg::this_grid();

    attention_rms_norm(p);
    grid.sync();
    qkv_rope_cache(p);
    grid.sync();
    sliding_window_attention(p);
    grid.sync();
    attention_output_residual_rms(p);
    grid.sync();
    post_attention_norm_and_router(p);
    grid.sync();
    moe_gate_up(p);
    grid.sync();
    moe_down_residual_rms(p);
}

// Complete full-context decode layer. The producer writes flash-style partials
// for all heads and partitions; the cooperative grid combines directly to
// FP16 and continues through the shared output/MoE stages.
__launch_bounds__(block_size, 1) __global__ void gptoss_decode_layer_full_kernel(gptoss_decode_layer_params p) {
    cg::grid_group grid = cg::this_grid();

    attention_rms_norm(p);
    grid.sync();
    qkv_rope_cache(p);
    grid.sync();
    full_attention_parts(p);
    grid.sync();
    full_attention_combine(p);
    grid.sync();
    attention_output_residual_rms(p);
    grid.sync();
    post_attention_norm_and_router(p);
    grid.sync();
    moe_gate_up(p);
    grid.sync();
    moe_down_residual_rms(p);
}
