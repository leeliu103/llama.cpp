#include "gptoss-kernel-hip.h"

#include "gptoss-config.h"

#include <hip/hip_runtime.h>

__global__ void gptoss_embedding_q8_0_kernel(float *, const uint8_t *, const int32_t *);
__global__ void gptoss_rms_norm_f16_kernel(const float *, const float *, __half *, float);
__global__ void gptoss_post_attention_rms_norm_f32_f16_kernel(const float *, const float *, float *, __half *, float);
__global__ void gptoss_build_rope_cache_f32(float *         cache,
                                            const int32_t * positions,
                                            uint32_t        n_tokens,
                                            float           freq_scale,
                                            float           ext_factor,
                                            float           attn_factor,
                                            float           corr_low,
                                            float           corr_high,
                                            float           theta_scale);
__global__ void gptoss_qkv_rope_cache_f16(__half *,
                                          __half *,
                                          __half *,
                                          const __half *,
                                          const float *,
                                          const int64_t *);
__global__ void gptoss_biased_topk_softmax_kernel(const float *, const float *, int32_t *, float *, uint32_t);
__global__ void gptoss_ogs_build_routes(const int32_t *,
                                        int32_t *,
                                        int32_t *,
                                        int32_t *,
                                        int32_t *,
                                        int32_t *,
                                        int32_t *,
                                        uint32_t,
                                        uint32_t);
__global__ void gptoss_moe_combine_residual_f32(float *, const float *, const float *, const float *, uint32_t);
__global__ void gptoss_lm_head_mmvq_q8_0_f16_kernel(const uint8_t *, const __half *, float *);
__global__ void gptoss_decode_layer_swa_kernel(gptoss_decode_layer_params);
__global__ void gptoss_decode_layer_full_kernel(gptoss_decode_layer_params);

hipError_t gptoss_embedding_q8_0_launch(float *         output,
                                        const uint8_t * weight,
                                        const int32_t * tokens,
                                        uint32_t        n_tokens,
                                        hipStream_t     stream) {
    hipLaunchKernelGGL(gptoss_embedding_q8_0_kernel, dim3(n_tokens), dim3(256), 0, stream, output, weight, tokens);
    return hipGetLastError();
}

hipError_t gptoss_rms_norm_launch(const float * input,
                                  const float * weight,
                                  __half *      output,
                                  float         eps,
                                  uint32_t      n_tokens,
                                  hipStream_t   stream) {
    hipLaunchKernelGGL(gptoss_rms_norm_f16_kernel, dim3(n_tokens), dim3(1024), 0, stream, input, weight, output, eps);
    return hipGetLastError();
}

hipError_t gptoss_post_attention_rms_norm_launch(const float * input,
                                                 const float * weight,
                                                 float *       output_f32,
                                                 __half *      output_f16,
                                                 float         eps,
                                                 uint32_t      n_tokens,
                                                 hipStream_t   stream) {
    hipLaunchKernelGGL(gptoss_post_attention_rms_norm_f32_f16_kernel, dim3(n_tokens), dim3(1024), 0, stream, input,
                       weight, output_f32, output_f16, eps);
    return hipGetLastError();
}

hipError_t gptoss_build_rope_cache_launch(float *               cache,
                                          const int32_t *       positions,
                                          uint32_t              n_tokens,
                                          float                 freq_scale,
                                          float                 ext_factor,
                                          float                 attn_factor,
                                          float                 corr_low,
                                          float                 corr_high,
                                          float                 theta_scale,
                                          hipStream_t           stream) {
    constexpr uint32_t tokens_per_block = 8;
    const dim3         block(gptoss_head_size / 2, tokens_per_block);
    const dim3         grid((n_tokens + tokens_per_block - 1) / tokens_per_block);
    hipLaunchKernelGGL(gptoss_build_rope_cache_f32, grid, block, 0, stream, cache, positions, n_tokens, freq_scale,
                       ext_factor, attn_factor, corr_low, corr_high, theta_scale);
    return hipGetLastError();
}

hipError_t gptoss_qkv_rope_cache_launch(__half *        q,
                                        __half *        cache_k,
                                        __half *        cache_v,
                                        const __half *  qkv,
                                        const float *   rope_cache,
                                        const int64_t * kv_dst_rows,
                                        uint32_t        n_tokens,
                                        hipStream_t     stream) {
    hipLaunchKernelGGL(gptoss_qkv_rope_cache_f16,
                       dim3(n_tokens, gptoss_query_head_count / gptoss_kv_head_count + 1), dim3(256), 0, stream, q,
                       cache_k, cache_v, qkv, rope_cache, kv_dst_rows);
    return hipGetLastError();
}

hipError_t gptoss_biased_topk_softmax_launch(const float * router_logits,
                                             const float * router_bias,
                                             int32_t *     selected_ids,
                                             float *       selected_weights,
                                             uint32_t      n_tokens,
                                             hipStream_t   stream) {
    hipLaunchKernelGGL(gptoss_biased_topk_softmax_kernel, dim3((n_tokens + 3) / 4), dim3(32, 4), 0, stream,
                       router_logits, router_bias, selected_ids, selected_weights, n_tokens);
    return hipGetLastError();
}

hipError_t gptoss_ogs_build_routes_launch(const int32_t * expert_ids,
                                          int32_t *       gather_token_indices,
                                          int32_t *       scatter_route_indices,
                                          int32_t *       expert_counts,
                                          int32_t *       route_offsets,
                                          int32_t *       block_offsets,
                                          int32_t *       block_schedule,
                                          uint32_t        route_count,
                                          uint32_t        schedule_capacity,
                                          hipStream_t     stream) {
    hipLaunchKernelGGL(gptoss_ogs_build_routes, dim3(1), dim3(256), 0, stream, expert_ids, gather_token_indices,
                       scatter_route_indices, expert_counts, route_offsets, block_offsets, block_schedule, route_count,
                       schedule_capacity);
    return hipGetLastError();
}

hipError_t gptoss_moe_combine_launch(float *       output,
                                     const float * residual,
                                     const float * expert_outputs,
                                     const float * routing_weights,
                                     uint32_t      n_tokens,
                                     hipStream_t   stream) {
    const uint32_t blocks = (n_tokens * gptoss_hidden_size + 255) / 256;
    hipLaunchKernelGGL(gptoss_moe_combine_residual_f32, dim3(blocks), dim3(256), 0, stream, output, residual,
                       expert_outputs, routing_weights, n_tokens);
    return hipGetLastError();
}

hipError_t gptoss_lm_head_mmvq_launch(const uint8_t * weight,
                                      const __half *  activation,
                                      float *         logits,
                                      hipStream_t     stream) {
    hipLaunchKernelGGL(gptoss_lm_head_mmvq_q8_0_f16_kernel, dim3(gptoss_vocabulary_size / 4), dim3(32, 4), 0, stream,
                       weight, activation, logits);
    return hipGetLastError();
}

hipError_t gptoss_decode_layer_launch(bool swa, const gptoss_decode_layer_params & params, hipStream_t stream) {
    const void * kernel = swa ? reinterpret_cast<const void *>(gptoss_decode_layer_swa_kernel) :
                                reinterpret_cast<const void *>(gptoss_decode_layer_full_kernel);

    void * kernel_params[] = { const_cast<gptoss_decode_layer_params *>(&params) };
    return hipLaunchCooperativeKernel(kernel, dim3(gptoss_decode_grid_blocks),
                                      dim3(gptoss_decode_block_x, gptoss_decode_block_y), kernel_params, 0, stream);
}
