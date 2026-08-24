#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>

hipError_t gptoss_embedding_q8_0_launch(float *         output,
                                        const uint8_t * weight,
                                        const int32_t * tokens,
                                        uint32_t        n_tokens,
                                        hipStream_t     stream);

hipError_t gptoss_rms_norm_launch(const float * input,
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
                                          float                 freq_scale,
                                          float                 ext_factor,
                                          float                 attn_factor,
                                          float                 corr_low,
                                          float                 corr_high,
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

hipError_t gptoss_lm_head_mmvq_launch(const uint8_t * weight,
                                      const __half *  activation,
                                      float *         logits,
                                      hipStream_t     stream);
