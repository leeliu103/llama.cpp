#include <float.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace cg = cooperative_groups;

struct gptoss_decode_layer_params {
    float *       next;
    const float * current;

    float *   norm_scale;
    __half *  activation;
    float *   router_scores;
    int32_t * selected_experts;
    float *   selected_weights;

    __half *        cache_k;
    __half *        cache_v;
    const int32_t * kv_rows;
    uint32_t        n_kv;
    uint32_t        kv_write_row;

    float *  attention_parts;
    float2 * attention_meta;

    const float *  attn_norm;
    const int8_t * qkv_values;
    const __half * qkv_scales;
    const float *  attn_q_bias;
    const float *  attn_k_bias;
    const float *  attn_v_bias;

    const int8_t * attn_output_values;
    const __half * attn_output_scales;
    const float *  attn_output_bias;
    const float *  attn_sinks;

    const float * post_attn_norm;
    const float * router_weight;
    const float * router_bias;

    const uint8_t * moe_down_values;
    const uint8_t * moe_down_scales;
    const uint8_t * moe_gate_up_values;
    const uint8_t * moe_gate_up_scales;
    const float *   moe_down_bias;
    const float *   moe_gate_up_bias;

    int32_t position;
    float   rms_epsilon;
    float   rope_freq_scale;
    float   rope_ext_factor;
    float   rope_attn_factor;
    float   rope_corr_low;
    float   rope_corr_high;
    float   rope_theta_scale;
};

namespace {

constexpr uint32_t warp_size        = 32;
constexpr uint32_t warps_per_block  = 8;
constexpr uint32_t block_size       = warp_size * warps_per_block;
constexpr uint32_t hidden_size      = 2880;
constexpr uint32_t padded_hidden    = 3072;
constexpr uint32_t q_head_count     = 64;
constexpr uint32_t kv_head_count    = 8;
constexpr uint32_t head_size        = 64;
constexpr uint32_t q_size           = q_head_count * head_size;
constexpr uint32_t kv_size          = kv_head_count * head_size;
constexpr uint32_t expert_count     = 32;
constexpr uint32_t experts_used     = 4;
constexpr uint32_t expert_size      = hidden_size;
constexpr uint32_t quant_block_size = 32;
constexpr uint32_t swa_size         = 128;
constexpr uint32_t attention_tile   = 128;

constexpr uint32_t qkv_blocks_per_row        = hidden_size / quant_block_size;
constexpr uint32_t output_blocks_per_row     = q_size / quant_block_size;
constexpr uint32_t mxfp4_input_blocks        = hidden_size / quant_block_size;
constexpr uint32_t mxfp4_value_bytes_per_row = padded_hidden / 2;
constexpr uint32_t mxfp4_scales_per_row      = padded_hidden / quant_block_size;
constexpr uint32_t gate_up_rows_per_expert   = 2 * padded_hidden;
constexpr uint32_t down_rows_per_expert      = padded_hidden;

__device__ __forceinline__ uint32_t lane_id() {
    return threadIdx.x & (warp_size - 1);
}

__device__ __forceinline__ uint32_t warp_id() {
    return threadIdx.x / warp_size;
}

__device__ __forceinline__ uint32_t global_warp_id() {
    return blockIdx.x * warps_per_block + warp_id();
}

__device__ __forceinline__ uint32_t global_warp_count() {
    return gridDim.x * warps_per_block;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, warp_size);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor(value, offset, warp_size));
    }
    return value;
}

__device__ float block_sum(float value, float * warp_sums) {
    const uint32_t lane = lane_id();
    const uint32_t warp = warp_id();

    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = lane < warps_per_block ? warp_sums[lane] : 0.0f;
    return warp_sum(value);
}

__device__ __forceinline__ float round_f16(float value) {
    return __half2float(__float2half_rn(value));
}

__device__ __forceinline__ float2 load_f16x2(const __half * values) {
    return __half22float2(*reinterpret_cast<const __half2 *>(values));
}

__device__ __forceinline__ float4 warp_sum(float4 value) {
    value.x = warp_sum(value.x);
    value.y = warp_sum(value.y);
    value.z = warp_sum(value.z);
    value.w = warp_sum(value.w);
    return value;
}

__device__ __forceinline__ float2 warp_sum(float2 value) {
    value.x = warp_sum(value.x);
    value.y = warp_sum(value.y);
    return value;
}

__device__ float4 q8_0_dot4(const int8_t * values,
                            const __half * scales,
                            uint32_t       row0,
                            uint32_t       row1,
                            uint32_t       row2,
                            uint32_t       row3,
                            uint32_t       columns,
                            uint32_t       blocks_per_row,
                            const __half * activation) {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t block = lane_id(); block < blocks_per_row; block += warp_size) {
        const uint32_t column = block * quant_block_size;
        const int8_t * w0     = values + static_cast<uint64_t>(row0) * columns + column;
        const int8_t * w1     = values + static_cast<uint64_t>(row1) * columns + column;
        const int8_t * w2     = values + static_cast<uint64_t>(row2) * columns + column;
        const int8_t * w3     = values + static_cast<uint64_t>(row3) * columns + column;

        float4 block_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (uint32_t i = 0; i < quant_block_size; i += 2) {
            const float2 x = load_f16x2(activation + column + i);
            block_sum.x    = fmaf(static_cast<float>(w0[i]), x.x, block_sum.x);
            block_sum.x    = fmaf(static_cast<float>(w0[i + 1]), x.y, block_sum.x);
            block_sum.y    = fmaf(static_cast<float>(w1[i]), x.x, block_sum.y);
            block_sum.y    = fmaf(static_cast<float>(w1[i + 1]), x.y, block_sum.y);
            block_sum.z    = fmaf(static_cast<float>(w2[i]), x.x, block_sum.z);
            block_sum.z    = fmaf(static_cast<float>(w2[i + 1]), x.y, block_sum.z);
            block_sum.w    = fmaf(static_cast<float>(w3[i]), x.x, block_sum.w);
            block_sum.w    = fmaf(static_cast<float>(w3[i + 1]), x.y, block_sum.w);
        }

        sum.x = fmaf(__half2float(scales[static_cast<uint64_t>(row0) * blocks_per_row + block]), block_sum.x, sum.x);
        sum.y = fmaf(__half2float(scales[static_cast<uint64_t>(row1) * blocks_per_row + block]), block_sum.y, sum.y);
        sum.z = fmaf(__half2float(scales[static_cast<uint64_t>(row2) * blocks_per_row + block]), block_sum.z, sum.z);
        sum.w = fmaf(__half2float(scales[static_cast<uint64_t>(row3) * blocks_per_row + block]), block_sum.w, sum.w);
    }

    return warp_sum(sum);
}

__device__ __forceinline__ float e8m0_scale(uint8_t scale) {
    const uint32_t encoded = static_cast<uint32_t>(scale) << 23;
    return __uint_as_float(encoded < 0x00400000u ? 0x00400000u : encoded);
}

__device__ __forceinline__ float mxfp4_value(uint8_t code) {
    const uint32_t magnitude = code & 7u;
    float          value;

    if (magnitude < 4u) {
        value = 0.5f * static_cast<float>(magnitude);
    } else {
        value = static_cast<float>(2u + (magnitude & 1u)) * static_cast<float>(1u << ((magnitude - 4u) / 2u));
    }

    return code & 8u ? -value : value;
}

__device__ float4 mxfp4_dot4(const uint8_t * values,
                             const uint8_t * scales,
                             uint32_t        row0,
                             uint32_t        row1,
                             uint32_t        row2,
                             uint32_t        row3,
                             const __half *  activation) {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t block = lane_id(); block < mxfp4_input_blocks; block += warp_size) {
        const uint32_t  byte = block * (quant_block_size / 2);
        const uint8_t * w0   = values + static_cast<uint64_t>(row0) * mxfp4_value_bytes_per_row + byte;
        const uint8_t * w1   = values + static_cast<uint64_t>(row1) * mxfp4_value_bytes_per_row + byte;
        const uint8_t * w2   = values + static_cast<uint64_t>(row2) * mxfp4_value_bytes_per_row + byte;
        const uint8_t * w3   = values + static_cast<uint64_t>(row3) * mxfp4_value_bytes_per_row + byte;

        float4 block_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (uint32_t i = 0; i < quant_block_size / 2; ++i) {
            const float2  x  = load_f16x2(activation + block * quant_block_size + 2 * i);
            const uint8_t q0 = w0[i];
            const uint8_t q1 = w1[i];
            const uint8_t q2 = w2[i];
            const uint8_t q3 = w3[i];

            block_sum.x = fmaf(mxfp4_value(q0 & 15u), x.x, block_sum.x);
            block_sum.x = fmaf(mxfp4_value(q0 >> 4), x.y, block_sum.x);
            block_sum.y = fmaf(mxfp4_value(q1 & 15u), x.x, block_sum.y);
            block_sum.y = fmaf(mxfp4_value(q1 >> 4), x.y, block_sum.y);
            block_sum.z = fmaf(mxfp4_value(q2 & 15u), x.x, block_sum.z);
            block_sum.z = fmaf(mxfp4_value(q2 >> 4), x.y, block_sum.z);
            block_sum.w = fmaf(mxfp4_value(q3 & 15u), x.x, block_sum.w);
            block_sum.w = fmaf(mxfp4_value(q3 >> 4), x.y, block_sum.w);
        }

        sum.x =
            fmaf(e8m0_scale(scales[static_cast<uint64_t>(row0) * mxfp4_scales_per_row + block]), block_sum.x, sum.x);
        sum.y =
            fmaf(e8m0_scale(scales[static_cast<uint64_t>(row1) * mxfp4_scales_per_row + block]), block_sum.y, sum.y);
        sum.z =
            fmaf(e8m0_scale(scales[static_cast<uint64_t>(row2) * mxfp4_scales_per_row + block]), block_sum.z, sum.z);
        sum.w =
            fmaf(e8m0_scale(scales[static_cast<uint64_t>(row3) * mxfp4_scales_per_row + block]), block_sum.w, sum.w);
    }

    return warp_sum(sum);
}

__device__ float2
mxfp4_dot2(const uint8_t * values, const uint8_t * scales, uint32_t row0, uint32_t row1, const __half * activation) {
    float2 sum = make_float2(0.0f, 0.0f);

    for (uint32_t block = lane_id(); block < mxfp4_input_blocks; block += warp_size) {
        const uint32_t  byte = block * (quant_block_size / 2);
        const uint8_t * w0   = values + static_cast<uint64_t>(row0) * mxfp4_value_bytes_per_row + byte;
        const uint8_t * w1   = values + static_cast<uint64_t>(row1) * mxfp4_value_bytes_per_row + byte;

        float2 block_sum = make_float2(0.0f, 0.0f);
#pragma unroll
        for (uint32_t i = 0; i < quant_block_size / 2; ++i) {
            const float2  x  = load_f16x2(activation + block * quant_block_size + 2 * i);
            const uint8_t q0 = w0[i];
            const uint8_t q1 = w1[i];

            block_sum.x = fmaf(mxfp4_value(q0 & 15u), x.x, block_sum.x);
            block_sum.x = fmaf(mxfp4_value(q0 >> 4), x.y, block_sum.x);
            block_sum.y = fmaf(mxfp4_value(q1 & 15u), x.x, block_sum.y);
            block_sum.y = fmaf(mxfp4_value(q1 >> 4), x.y, block_sum.y);
        }

        sum.x =
            fmaf(e8m0_scale(scales[static_cast<uint64_t>(row0) * mxfp4_scales_per_row + block]), block_sum.x, sum.x);
        sum.y =
            fmaf(e8m0_scale(scales[static_cast<uint64_t>(row1) * mxfp4_scales_per_row + block]), block_sum.y, sum.y);
    }

    return warp_sum(sum);
}

__device__ void rms_norm(const float * input, const float * weight, __half * output, float epsilon) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ float warp_sums[warps_per_block];
    float            sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden_size; i += block_size) {
        const float value = input[i];
        sum += value * value;
    }
    sum = block_sum(sum, warp_sums);

    const float scale = rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
    for (uint32_t i = threadIdx.x; i < hidden_size; i += block_size) {
        output[i] = __float2half_rn(input[i] * scale * weight[i]);
    }
}

__device__ void rms_scale(const float * input, float * scale, float epsilon) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ float warp_sums[warps_per_block];
    float            sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden_size; i += block_size) {
        const float value = input[i];
        sum += value * value;
    }
    sum = block_sum(sum, warp_sums);

    if (threadIdx.x == 0) {
        scale[0] = rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
    }
}

__device__ __forceinline__ float rope_ramp(float low, float high, uint32_t pair) {
    const float value = (static_cast<float>(pair) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, value));
}

__device__ void apply_rope(const gptoss_decode_layer_params & p,
                           uint32_t                           pair,
                           float                              x0,
                           float                              x1,
                           float &                            y0,
                           float &                            y1) {
    const float theta_extrap = static_cast<float>(p.position) * powf(p.rope_theta_scale, static_cast<float>(pair));
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
    y0                 = fmaf(x0, cosine, -(x1 * sine));
    y1                 = fmaf(x1, cosine, x0 * sine);
}

__device__ void qkv_rope_cache(const gptoss_decode_layer_params & p) {
    constexpr uint32_t pairs_per_head      = head_size / 2;
    constexpr uint32_t pair_quads_per_head = pairs_per_head / 2;
    constexpr uint32_t q_tasks             = q_head_count * pair_quads_per_head;
    constexpr uint32_t k_tasks             = kv_head_count * pair_quads_per_head;
    constexpr uint32_t v_tasks             = kv_size / 4;
    constexpr uint32_t qk_tasks            = q_tasks + k_tasks;
    constexpr uint32_t total_tasks         = qk_tasks + v_tasks;

    const __half * input      = p.activation;
    __half *       q          = p.activation + hidden_size;
    const uint64_t cache_base = static_cast<uint64_t>(p.kv_write_row) * kv_size;

    for (uint32_t task = global_warp_id(); task < total_tasks; task += global_warp_count()) {
        if (task < qk_tasks) {
            const bool     is_query  = task < q_tasks;
            const uint32_t pair_quad = is_query ? task : task - q_tasks;
            const uint32_t head      = pair_quad / pair_quads_per_head;
            const uint32_t pair0     = 2 * (pair_quad % pair_quads_per_head);
            const uint32_t local0    = head * head_size + pair0;
            const uint32_t local1    = local0 + pairs_per_head;
            const uint32_t row_base  = is_query ? 0 : q_size;

            const float4 dot =
                q8_0_dot4(p.qkv_values, p.qkv_scales, row_base + local0, row_base + local1, row_base + local0 + 1,
                          row_base + local1 + 1, hidden_size, qkv_blocks_per_row, input);

            if (lane_id() < 2) {
                const uint32_t which = lane_id();
                const uint32_t pair  = pair0 + which;
                const uint32_t low   = local0 + which;
                const uint32_t high  = local1 + which;
                const float *  bias  = is_query ? p.attn_q_bias : p.attn_k_bias;
                const float    x0    = round_f16((which == 0 ? dot.x : dot.z) + bias[low]);
                const float    x1    = round_f16((which == 0 ? dot.y : dot.w) + bias[high]);
                float          y0;
                float          y1;
                apply_rope(p, pair, x0, x1, y0, y1);

                if (is_query) {
                    q[low]  = __float2half_rn(y0 * 0.125f);
                    q[high] = __float2half_rn(y1 * 0.125f);
                } else {
                    p.cache_k[cache_base + low]  = __float2half_rn(y0);
                    p.cache_k[cache_base + high] = __float2half_rn(y1);
                }
            }
        } else {
            const uint32_t row        = 4 * (task - qk_tasks);
            const uint32_t weight_row = q_size + kv_size + row;
            const float4   dot = q8_0_dot4(p.qkv_values, p.qkv_scales, weight_row, weight_row + 1, weight_row + 2,
                                           weight_row + 3, hidden_size, qkv_blocks_per_row, input);

            if (lane_id() == 0) {
                p.cache_v[cache_base + row]     = __float2half_rn(dot.x + p.attn_v_bias[row]);
                p.cache_v[cache_base + row + 1] = __float2half_rn(dot.y + p.attn_v_bias[row + 1]);
                p.cache_v[cache_base + row + 2] = __float2half_rn(dot.z + p.attn_v_bias[row + 2]);
                p.cache_v[cache_base + row + 3] = __float2half_rn(dot.w + p.attn_v_bias[row + 3]);
            }
        }
    }
}

__device__ void sliding_window_attention(const gptoss_decode_layer_params & p) {
    __half * q = p.activation + hidden_size;

    for (uint32_t head = global_warp_id(); head < q_head_count; head += global_warp_count()) {
        const uint32_t kv_head = head / (q_head_count / kv_head_count);
        const __half * query   = q + static_cast<uint64_t>(head) * head_size;

        float scores[swa_size / warp_size];
        float weights[swa_size / warp_size];
        float local_max = lane_id() == 0 ? p.attn_sinks[head] : -FLT_MAX;

#pragma unroll
        for (uint32_t group = 0; group < swa_size / warp_size; ++group) {
            const uint32_t key   = group * warp_size + lane_id();
            float          score = -FLT_MAX;
            if (key < p.n_kv) {
                const uint32_t row = static_cast<uint32_t>(p.kv_rows[key]);
                const __half * k   = p.cache_k + static_cast<uint64_t>(row) * kv_size + kv_head * head_size;
                score              = 0.0f;
#pragma unroll
                for (uint32_t d = 0; d < head_size; d += 2) {
                    const float2 q2 = load_f16x2(query + d);
                    const float2 k2 = load_f16x2(k + d);
                    score           = fmaf(q2.x, k2.x, score);
                    score           = fmaf(q2.y, k2.y, score);
                }
            }
            scores[group] = score;
            local_max     = fmaxf(local_max, score);
        }

        const float maximum     = warp_max(local_max);
        float       denominator = lane_id() == 0 ? expf(p.attn_sinks[head] - maximum) : 0.0f;
#pragma unroll
        for (uint32_t group = 0; group < swa_size / warp_size; ++group) {
            const uint32_t key    = group * warp_size + lane_id();
            const float    weight = key < p.n_kv ? expf(scores[group] - maximum) : 0.0f;
            weights[group]        = weight;
            denominator += weight;
        }
        denominator = warp_sum(denominator);

        float output0 = 0.0f;
        float output1 = 0.0f;
        for (uint32_t key = 0; key < p.n_kv; ++key) {
            const uint32_t owner  = key & (warp_size - 1);
            const uint32_t group  = key / warp_size;
            const float    weight = __shfl(weights[group], owner, warp_size);
            const uint32_t row    = static_cast<uint32_t>(p.kv_rows[key]);
            const __half * v      = p.cache_v + static_cast<uint64_t>(row) * kv_size + kv_head * head_size;
            output0               = fmaf(weight, __half2float(v[lane_id()]), output0);
            output1               = fmaf(weight, __half2float(v[lane_id() + warp_size]), output1);
        }

        const float inverse                                                = 1.0f / denominator;
        q[static_cast<uint64_t>(head) * head_size + lane_id()]             = __float2half_rn(output0 * inverse);
        q[static_cast<uint64_t>(head) * head_size + lane_id() + warp_size] = __float2half_rn(output1 * inverse);
    }
}

__device__ void full_attention_parts(const gptoss_decode_layer_params & p) {
    const uint32_t partition_count = (p.n_kv + attention_tile - 1) / attention_tile;
    const uint32_t task_count      = q_head_count * partition_count;
    const __half * q               = p.activation + hidden_size;

    for (uint32_t task = global_warp_id(); task < task_count; task += global_warp_count()) {
        const uint32_t head      = task / partition_count;
        const uint32_t partition = task % partition_count;
        const uint32_t kv_head   = head / (q_head_count / kv_head_count);
        const uint32_t begin     = partition * attention_tile;
        const uint32_t remaining = p.n_kv - begin;
        const uint32_t count     = remaining < attention_tile ? remaining : attention_tile;
        const __half * query     = q + static_cast<uint64_t>(head) * head_size;

        float scores[attention_tile / warp_size];
        float weights[attention_tile / warp_size];
        float local_max = -FLT_MAX;

#pragma unroll
        for (uint32_t group = 0; group < attention_tile / warp_size; ++group) {
            const uint32_t key_in_part = group * warp_size + lane_id();
            float          score       = -FLT_MAX;
            if (key_in_part < count) {
                const uint32_t row = static_cast<uint32_t>(p.kv_rows[begin + key_in_part]);
                const __half * k   = p.cache_k + static_cast<uint64_t>(row) * kv_size + kv_head * head_size;
                score              = 0.0f;
#pragma unroll
                for (uint32_t d = 0; d < head_size; d += 2) {
                    const float2 q2 = load_f16x2(query + d);
                    const float2 k2 = load_f16x2(k + d);
                    score           = fmaf(q2.x, k2.x, score);
                    score           = fmaf(q2.y, k2.y, score);
                }
            }
            scores[group] = score;
            local_max     = fmaxf(local_max, score);
        }

        const float maximum     = warp_max(local_max);
        float       denominator = 0.0f;
#pragma unroll
        for (uint32_t group = 0; group < attention_tile / warp_size; ++group) {
            const uint32_t key_in_part = group * warp_size + lane_id();
            const float    weight      = key_in_part < count ? expf(scores[group] - maximum) : 0.0f;
            weights[group]             = weight;
            denominator += weight;
        }
        denominator = warp_sum(denominator);

        float output0 = 0.0f;
        float output1 = 0.0f;
        for (uint32_t key_in_part = 0; key_in_part < count; ++key_in_part) {
            const uint32_t owner  = key_in_part & (warp_size - 1);
            const uint32_t group  = key_in_part / warp_size;
            const float    weight = __shfl(weights[group], owner, warp_size);
            const uint32_t row    = static_cast<uint32_t>(p.kv_rows[begin + key_in_part]);
            const __half * v      = p.cache_v + static_cast<uint64_t>(row) * kv_size + kv_head * head_size;
            output0               = fmaf(weight, __half2float(v[lane_id()]), output0);
            output1               = fmaf(weight, __half2float(v[lane_id() + warp_size]), output1);
        }

        const uint64_t part_base = (static_cast<uint64_t>(head) * partition_count + partition) * head_size;
        p.attention_parts[part_base + lane_id()]             = output0;
        p.attention_parts[part_base + lane_id() + warp_size] = output1;
        if (lane_id() == 0) {
            p.attention_meta[static_cast<uint64_t>(head) * partition_count + partition] =
                make_float2(maximum, denominator);
        }
    }
}

__device__ void full_attention_combine(const gptoss_decode_layer_params & p) {
    const uint32_t partition_count = (p.n_kv + attention_tile - 1) / attention_tile;
    __half *       q               = p.activation + hidden_size;

    for (uint32_t head = global_warp_id(); head < q_head_count; head += global_warp_count()) {
        float local_max = lane_id() == 0 ? p.attn_sinks[head] : -FLT_MAX;
        for (uint32_t partition = lane_id(); partition < partition_count; partition += warp_size) {
            const float2 meta = p.attention_meta[static_cast<uint64_t>(head) * partition_count + partition];
            local_max         = fmaxf(local_max, meta.x);
        }
        const float maximum = warp_max(local_max);

        float local_denominator = lane_id() == 0 ? expf(p.attn_sinks[head] - maximum) : 0.0f;
        for (uint32_t partition = lane_id(); partition < partition_count; partition += warp_size) {
            const float2 meta = p.attention_meta[static_cast<uint64_t>(head) * partition_count + partition];
            local_denominator = fmaf(expf(meta.x - maximum), meta.y, local_denominator);
        }
        const float denominator = warp_sum(local_denominator);

        float output0 = 0.0f;
        float output1 = 0.0f;
        for (uint32_t base = 0; base < partition_count; base += warp_size) {
            const uint32_t partition = base + lane_id();
            float          scale     = 0.0f;
            if (partition < partition_count) {
                const float2 meta = p.attention_meta[static_cast<uint64_t>(head) * partition_count + partition];
                scale             = expf(meta.x - maximum);
            }

#pragma unroll
            for (uint32_t owner = 0; owner < warp_size; ++owner) {
                const uint32_t source_partition = base + owner;
                if (source_partition < partition_count) {
                    const float    source_scale = __shfl(scale, owner, warp_size);
                    const uint64_t part_base =
                        (static_cast<uint64_t>(head) * partition_count + source_partition) * head_size;
                    output0 = fmaf(source_scale, p.attention_parts[part_base + lane_id()], output0);
                    output1 = fmaf(source_scale, p.attention_parts[part_base + lane_id() + warp_size], output1);
                }
            }
        }

        const float inverse                                                = 1.0f / denominator;
        q[static_cast<uint64_t>(head) * head_size + lane_id()]             = __float2half_rn(output0 * inverse);
        q[static_cast<uint64_t>(head) * head_size + lane_id() + warp_size] = __float2half_rn(output1 * inverse);
    }
}

__device__ void attention_output(const gptoss_decode_layer_params & p) {
    constexpr uint32_t rows_per_task = 4;
    constexpr uint32_t task_count    = hidden_size / rows_per_task;
    const __half *     input         = p.activation + hidden_size;

    for (uint32_t task = global_warp_id(); task < task_count; task += global_warp_count()) {
        const uint32_t row = rows_per_task * task;
        const float4 dot = q8_0_dot4(p.attn_output_values, p.attn_output_scales, row, row + 1, row + 2, row + 3, q_size,
                                     output_blocks_per_row, input);

        if (lane_id() < rows_per_task) {
            const uint32_t output_row = row + lane_id();
            const float    value = lane_id() == 0 ? dot.x : lane_id() == 1 ? dot.y : lane_id() == 2 ? dot.z : dot.w;
            p.next[output_row]   = p.current[output_row] + (value + p.attn_output_bias[output_row]);
        }
    }
}

__device__ void post_attention_norm_and_router(const gptoss_decode_layer_params & p) {
    const uint32_t thread = blockIdx.x * block_size + threadIdx.x;
    const uint32_t stride = gridDim.x * block_size;
    const float    scale  = p.norm_scale[0];

    for (uint32_t i = thread; i < hidden_size; i += stride) {
        p.activation[i] = __float2half_rn(p.next[i] * scale * p.post_attn_norm[i]);
    }

    if (blockIdx.x >= expert_count) {
        return;
    }

    __shared__ float warp_sums[warps_per_block];
    float            router = 0.0f;
    const float *    weight = p.router_weight + static_cast<uint64_t>(blockIdx.x) * hidden_size;
    for (uint32_t i = threadIdx.x; i < hidden_size; i += block_size) {
        const float normalized = p.next[i] * scale * p.post_attn_norm[i];
        router                 = fmaf(normalized, weight[i], router);
    }
    router = block_sum(router, warp_sums);

    if (threadIdx.x == 0) {
        p.router_scores[blockIdx.x] = router + p.router_bias[blockIdx.x];
    }
}

__device__ void select_experts(const gptoss_decode_layer_params & p, int32_t * block_experts, float * block_logits) {
    if (warp_id() == 0) {
        float score = p.router_scores[lane_id()];
        if (isnan(score)) {
            score = -FLT_MAX;
        }

#pragma unroll
        for (uint32_t selected = 0; selected < experts_used; ++selected) {
            float    best_score  = score;
            uint32_t best_expert = lane_id();

#pragma unroll
            for (uint32_t offset = warp_size / 2; offset > 0; offset >>= 1) {
                const float    other_score  = __shfl_xor(best_score, offset, warp_size);
                const uint32_t other_expert = __shfl_xor(best_expert, offset, warp_size);
                if (other_score > best_score || (other_score == best_score && other_expert < best_expert)) {
                    best_score  = other_score;
                    best_expert = other_expert;
                }
            }

            if (lane_id() == 0) {
                block_experts[selected] = static_cast<int32_t>(best_expert);
                block_logits[selected]  = best_score;
            }
            if (lane_id() == best_expert) {
                score = -INFINITY;
            }
        }

        if (lane_id() == 0 && blockIdx.x == 0) {
            float maximum = block_logits[0];
#pragma unroll
            for (uint32_t i = 1; i < experts_used; ++i) {
                maximum = fmaxf(maximum, block_logits[i]);
            }

            float sum = 0.0f;
            float weights[experts_used];
#pragma unroll
            for (uint32_t i = 0; i < experts_used; ++i) {
                weights[i] = expf(block_logits[i] - maximum);
                sum += weights[i];
            }
#pragma unroll
            for (uint32_t i = 0; i < experts_used; ++i) {
                p.selected_experts[i] = block_experts[i];
                p.selected_weights[i] = weights[i] / sum;
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float swiglu_oai(float gate, float up) {
    gate = fminf(gate, 7.0f);
    up   = fmaxf(fminf(up, 7.0f), -7.0f);
    return gate / (1.0f + expf(-1.702f * gate)) * (1.0f + up);
}

__device__ void moe_gate_up(const gptoss_decode_layer_params & p) {
    __shared__ int32_t block_experts[experts_used];
    __shared__ float   block_logits[experts_used];
    select_experts(p, block_experts, block_logits);

    constexpr uint32_t row_pairs_per_expert = expert_size / 2;
    constexpr uint32_t task_count           = experts_used * row_pairs_per_expert;
    const __half *     input                = p.activation;
    __half *           output               = p.activation + hidden_size;

    for (uint32_t task = global_warp_id(); task < task_count; task += global_warp_count()) {
        const uint32_t slot       = task / row_pairs_per_expert;
        const uint32_t pair       = task % row_pairs_per_expert;
        const uint32_t row0       = 2 * pair;
        const uint32_t expert     = static_cast<uint32_t>(block_experts[slot]);
        const uint32_t expert_row = expert * gate_up_rows_per_expert;
        const uint32_t gate0      = expert_row + 2 * row0;
        const uint32_t up0        = gate0 + 1;
        const uint32_t gate1      = gate0 + 2;
        const uint32_t up1        = gate0 + 3;

        const float4 dot = mxfp4_dot4(p.moe_gate_up_values, p.moe_gate_up_scales, gate0, up0, gate1, up1, input);
        if (lane_id() == 0) {
            const uint64_t bias_base   = (static_cast<uint64_t>(expert) * expert_size + row0) * 2;
            const float    gate_value0 = dot.x + p.moe_gate_up_bias[bias_base];
            const float    up_value0   = dot.y + p.moe_gate_up_bias[bias_base + 1];
            const float    gate_value1 = dot.z + p.moe_gate_up_bias[bias_base + 2];
            const float    up_value1   = dot.w + p.moe_gate_up_bias[bias_base + 3];
            const uint64_t output_base = static_cast<uint64_t>(slot) * expert_size + row0;
            output[output_base]        = __float2half_rn(swiglu_oai(gate_value0, up_value0));
            output[output_base + 1]    = __float2half_rn(swiglu_oai(gate_value1, up_value1));
        }
    }
}

__device__ void moe_down(const gptoss_decode_layer_params & p) {
    constexpr uint32_t task_count = hidden_size / 2;
    const __half *     activation = p.activation + hidden_size;

    for (uint32_t pair = global_warp_id(); pair < task_count; pair += global_warp_count()) {
        const uint32_t row0    = 2 * pair;
        float          output0 = 0.0f;
        float          output1 = 0.0f;

#pragma unroll
        for (uint32_t slot = 0; slot < experts_used; ++slot) {
            const uint32_t expert     = static_cast<uint32_t>(p.selected_experts[slot]);
            const uint32_t weight_row = expert * down_rows_per_expert + row0;
            const float2   dot        = mxfp4_dot2(p.moe_down_values, p.moe_down_scales, weight_row, weight_row + 1,
                                                   activation + static_cast<uint64_t>(slot) * expert_size);
            const uint64_t bias_base  = static_cast<uint64_t>(expert) * hidden_size + row0;
            output0                   = fmaf(p.selected_weights[slot], dot.x + p.moe_down_bias[bias_base], output0);
            output1                   = fmaf(p.selected_weights[slot], dot.y + p.moe_down_bias[bias_base + 1], output1);
        }

        if (lane_id() == 0) {
            p.next[row0] += output0;
            p.next[row0 + 1] += output1;
        }
    }
}

}  // namespace

__launch_bounds__(block_size, 1) __global__ void gptoss_decode_layer_swa_kernel(gptoss_decode_layer_params p) {
    cg::grid_group grid = cg::this_grid();

    rms_norm(p.current, p.attn_norm, p.activation, p.rms_epsilon);
    grid.sync();
    qkv_rope_cache(p);
    grid.sync();
    sliding_window_attention(p);
    grid.sync();
    attention_output(p);
    grid.sync();
    rms_scale(p.next, p.norm_scale, p.rms_epsilon);
    grid.sync();
    post_attention_norm_and_router(p);
    grid.sync();
    moe_gate_up(p);
    grid.sync();
    moe_down(p);
}

__launch_bounds__(block_size, 1) __global__ void gptoss_decode_layer_full_kernel(gptoss_decode_layer_params p) {
    cg::grid_group grid = cg::this_grid();

    rms_norm(p.current, p.attn_norm, p.activation, p.rms_epsilon);
    grid.sync();
    qkv_rope_cache(p);
    grid.sync();
    full_attention_parts(p);
    grid.sync();
    full_attention_combine(p);
    grid.sync();
    attention_output(p);
    grid.sync();
    rms_scale(p.next, p.norm_scale, p.rms_epsilon);
    grid.sync();
    post_attention_norm_and_router(p);
    grid.sync();
    moe_gate_up(p);
    grid.sync();
    moe_down(p);
}
