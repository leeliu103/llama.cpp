#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>

constexpr uint32_t gptoss_decode_grid_blocks = 120;

struct gptoss_rope_corr_dims {
    float low;
    float high;
};

struct gptoss_decode_layer_params {
    float *       next;
    const float * cur;
    float *       rms_partials;

    __half * activation_scratch;
    __half * query;
    float *  attn_parts;
    float2 * attn_meta;
    float *  router;
    int32_t * expert_ids;
    float *   expert_weights;

    __half *        cache_k;
    __half *        cache_v;
    const int32_t * kv_rows;

    const float *  attn_norm;
    const int8_t * qkv_values;
    const float *  attn_q_bias;
    const float *  attn_k_bias;
    const float *  attn_v_bias;

    const int8_t * attn_output_values;
    const float *  attn_output_bias;
    const float *  attn_sinks;

    const float * post_attention_norm;
    const float * router_weight;
    const float * router_bias;

    const uint8_t * moe_down_values;
    const uint8_t * moe_gate_up_values;
    const float *   moe_down_bias;
    const float *   moe_gate_up_bias;

    uint32_t n_kv;
    uint32_t kv_write_row;
    uint32_t attn_parallel_blocks;
    int32_t  position;
    float    rms_epsilon;
    float    rope_freq_scale;
    float    rope_ext_factor;
    float    rope_attn_factor;
    float    rope_corr_low;
    float    rope_corr_high;
    float    rope_theta_scale;
    uint32_t reuse_attention_rms;
};

static_assert(sizeof(gptoss_decode_layer_params) == 272);

hipError_t gptoss_embedding_q8_0_launch(float *         output,
                                        const uint8_t * weight,
                                        const int32_t * tokens,
                                        uint32_t        n_tokens,
                                        hipStream_t     stream);

hipError_t gptoss_attention_rms_norm_launch(const float * input,
                                            const float * weight,
                                            __half *      output,
                                            float         eps,
                                            uint32_t      n_tokens,
                                            hipStream_t   stream);

hipError_t gptoss_post_attention_rms_norm_launch(const float * input,
                                                 const float * weight,
                                                 float *       output_f32,
                                                 __half *      output_f16,
                                                 float         eps,
                                                 uint32_t      n_tokens,
                                                 hipStream_t   stream);

hipError_t gptoss_build_rope_cache_launch(float *               cache,
                                          const int32_t *       positions,
                                          uint32_t              n_tokens,
                                          int                   n_dims,
                                          float                 freq_scale,
                                          float                 ext_factor,
                                          float                 attn_factor,
                                          gptoss_rope_corr_dims corr_dims,
                                          float                 theta_scale,
                                          hipStream_t           stream);

hipError_t gptoss_qkv_rope_cache_launch(__half *        q,
                                        __half *        cache_k,
                                        __half *        cache_v,
                                        const __half *  qkv,
                                        const float *   rope_cache,
                                        const int64_t * kv_dst_rows,
                                        uint32_t        n_tokens,
                                        hipStream_t     stream);

hipError_t gptoss_biased_topk_softmax_launch(const float * router_logits,
                                             const float * router_bias,
                                             int32_t *     selected_ids,
                                             float *       selected_weights,
                                             uint32_t      n_tokens,
                                             hipStream_t   stream);

hipError_t gptoss_ogs_build_routes_launch(const int32_t * expert_ids,
                                          int32_t *       gather_token_indices,
                                          int32_t *       scatter_route_indices,
                                          int32_t *       expert_counts,
                                          int32_t *       route_offsets,
                                          int32_t *       block_offsets,
                                          int32_t *       block_schedule,
                                          uint32_t        route_count,
                                          uint32_t        schedule_capacity,
                                          hipStream_t     stream);

hipError_t gptoss_moe_combine_launch(float *       output,
                                     const float * residual,
                                     const float * expert_outputs,
                                     const float * routing_weights,
                                     uint32_t      n_tokens,
                                     hipStream_t   stream);

hipError_t gptoss_output_rms_norm_quantize_launch(const float *   hidden,
                                                  const float *   weight,
                                                  const int32_t * output_rows,
                                                  uint8_t *       output,
                                                  float           eps,
                                                  uint32_t        n_outputs,
                                                  hipStream_t     stream);

hipError_t gptoss_lm_head_mmvq_launch(const uint8_t * weight,
                                      const uint8_t * activation,
                                      float *         logits,
                                      uint32_t        n_outputs,
                                      hipStream_t     stream);

hipError_t gptoss_decode_layer_launch(bool swa, const gptoss_decode_layer_params & params, hipStream_t stream);
