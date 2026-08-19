#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t attention_head_count    = 64;
constexpr uint32_t attention_head_count_kv = 8;
constexpr uint32_t attention_head_size     = 64;

__device__ float rope_yarn_ramp(float low, float high, int dimension) {
    const float y = (dimension / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ void rope_yarn(float                 theta_extrap,
                          float                 freq_scale,
                          float                 corr_low,
                          float                 corr_high,
                          int64_t               dimension,
                          float                 ext_factor,
                          float                 mscale,
                          float &               cos_theta,
                          float &               sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float       theta        = theta_interp;

    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_low, corr_high, dimension) * ext_factor;
        theta                = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

}  // namespace

__global__ void gptoss_build_rope_cache_f32(float *               cache,
                                            const int32_t *       positions,
                                            uint32_t              n_tokens,
                                            float                 freq_scale,
                                            float                 ext_factor,
                                            float                 attn_factor,
                                            float                 corr_low,
                                            float                 corr_high,
                                            float                 theta_scale) {
    constexpr uint32_t pairs = attention_head_size / 2;
    const uint32_t pair  = threadIdx.x;
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;

    if (token >= n_tokens || pair >= pairs) {
        return;
    }

    const int32_t  position  = positions[token];
    const uint32_t cache_idx = token * pairs + pair;
    const int      dimension = static_cast<int>(2 * pair);
    const float    theta     = position * powf(theta_scale, dimension / 2.0f);

    float cos_theta;
    float sin_theta;
    rope_yarn(theta, freq_scale, corr_low, corr_high, dimension, ext_factor, attn_factor, cos_theta, sin_theta);

    cache[2 * cache_idx]     = cos_theta;
    cache[2 * cache_idx + 1] = sin_theta;
}

__launch_bounds__(256) __global__ void gptoss_qkv_rope_cache_f16(__half *        q,
                                                                 __half *        cache_k,
                                                                 uint16_t *      cache_v,
                                                                 const __half *  qkv,
                                                                 const float *   rope_cache,
                                                                 const int64_t * kv_dst_rows) {
    constexpr uint32_t half_head_size = attention_head_size / 2;
    constexpr uint32_t query_size     = attention_head_count * attention_head_size;
    constexpr uint32_t key_value_size = attention_head_count_kv * attention_head_size;
    constexpr uint32_t qkv_size       = query_size + 2 * key_value_size;
    constexpr uint32_t query_groups   = attention_head_count / 8;

    const uint32_t token     = blockIdx.x;
    const uint32_t group     = blockIdx.y;
    const uint32_t pair      = threadIdx.x & (half_head_size - 1);
    const uint32_t head_lane = threadIdx.x / half_head_size;
    const uint32_t head      = group * 8 + head_lane;

    const uint64_t qkv_row     = static_cast<uint64_t>(token) * qkv_size;
    const uint64_t rope_offset = static_cast<uint64_t>(token) * attention_head_size + 2 * pair;
    const float    cos_theta   = rope_cache[rope_offset];
    const float    sin_theta   = rope_cache[rope_offset + 1];

    if (group < query_groups) {
        const uint64_t src = qkv_row + static_cast<uint64_t>(head) * attention_head_size;
        const uint64_t dst =
            static_cast<uint64_t>(token) * query_size + static_cast<uint64_t>(head) * attention_head_size;

        const float x0 = __half2float(qkv[src + pair]);
        const float x1 = __half2float(qkv[src + pair + half_head_size]);
        float       y0 = fmaf(x0, cos_theta, -(x1 * sin_theta));
        float       y1 = fmaf(x1, cos_theta, x0 * sin_theta);

        asm volatile("" : "+v"(y0), "+v"(y1));
        q[dst + pair]                  = __float2half_rn(y0);
        q[dst + pair + half_head_size] = __float2half_rn(y1);
        return;
    }

    const uint32_t kv_head   = head_lane;
    const uint64_t src_k     = qkv_row + query_size + static_cast<uint64_t>(kv_head) * attention_head_size;
    const uint64_t src_v     = src_k + key_value_size;
    const uint64_t cache_row = static_cast<uint64_t>(kv_dst_rows[token]);
    const uint64_t dst       = cache_row * key_value_size + static_cast<uint64_t>(kv_head) * attention_head_size;

    const float x0 = __half2float(qkv[src_k + pair]);
    const float x1 = __half2float(qkv[src_k + pair + half_head_size]);
    float       y0 = fmaf(x0, cos_theta, -(x1 * sin_theta));
    float       y1 = fmaf(x1, cos_theta, x0 * sin_theta);

    asm volatile("" : "+v"(y0), "+v"(y1));
    cache_k[dst + pair]                  = __float2half_rn(y0);
    cache_k[dst + pair + half_head_size] = __float2half_rn(y1);

    const uint16_t * qkv_bits            = reinterpret_cast<const uint16_t *>(qkv);
    cache_v[dst + pair]                  = qkv_bits[src_v + pair];
    cache_v[dst + pair + half_head_size] = qkv_bits[src_v + pair + half_head_size];
}
