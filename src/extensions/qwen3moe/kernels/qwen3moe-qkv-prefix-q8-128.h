#pragma once
#include "ggml-backend.h"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// Internal decode-prefix API. The backend owns graph admission and token-sticky
// status reset/seal. Arithmetic retains the measured native128 approximation.
enum qwen3moe_qkv_prefix_error : uint32_t { QWEN3MOE_QKV_PREFIX_BAD_SLOT = 1, QWEN3MOE_QKV_PREFIX_NONFINITE = 2 };
struct qwen3moe_qkv_prefix_rope {
    float epsilon, theta_scale, freq_scale, ext_factor, attn_factor, corr_dims[2];
};
struct qwen3moe_qkv_prefix_q8_128_args {
    const void * wq; const void * wk; const void * wv;
    uint64_t q_stride, k_stride, v_stride;
    size_t q_weight_bytes, k_weight_bytes, v_weight_bytes;
    const float * input; const float * input_gamma;
    const float * q_gamma; const float * k_gamma;
    const int32_t * position;
    const int64_t * k_slot; const int64_t * v_slot;
    int q_heads, kv_heads, v_q6;
    float input_epsilon;
    qwen3moe_qkv_prefix_rope q_rope, k_rope;
    float * normalized; // Materialized diagnostics included in candidate timing.
    int8_t * xq; float * xd; // Native group128 activation bytes and FP32 scales.
    float * q_projected; float * k_projected; float * v_projected;
    float * q_output; float * k_rotated;
    ggml_fp16_t * k_cache; ggml_fp16_t * v_cache;
    uint64_t k_row_bytes, v_row_bytes;
    size_t k_cache_bytes, v_cache_bytes;
    int64_t cache_rows;
    uint32_t * status;
};

// Metadata/range checking only, before any launch. All output ranges must be
// mutually disjoint and disjoint from every live read-only prefix input.
bool qwen3moe_qkv_prefix_q8_128_supported(const qwen3moe_qkv_prefix_q8_128_args * args);
// Does not reset status. Existing token-sticky errors leave neutral Q and no
// new cache writes; the caller drains/seals the established collective sequence.
// No grid barrier or cooperative launch.
hipError_t qwen3moe_qkv_prefix_q8_128_enqueue(const qwen3moe_qkv_prefix_q8_128_args * args, hipStream_t stream);

