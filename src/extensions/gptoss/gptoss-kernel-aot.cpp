#include "gptoss-kernel-aot.h"

#include "gptoss-config.h"
#include "gptoss-kernel-hip.h"

namespace {

constexpr uint32_t gptoss_q8_block_m       = 64;
constexpr uint32_t gptoss_q8_block_n       = 128;
constexpr uint32_t gptoss_q8_threads       = 128;
constexpr uint32_t gptoss_q8_shared_memory = 16384;

constexpr uint32_t gptoss_router_block_m       = 8;
constexpr uint32_t gptoss_router_threads       = 128;
constexpr uint32_t gptoss_router_shared_memory = 2048;

constexpr uint32_t gptoss_fa_block_q       = 8;
constexpr uint32_t gptoss_fa_threads       = 128;
constexpr uint32_t gptoss_fa_shared_memory = 8192;

constexpr uint32_t gptoss_ogs_block_n       = 128;
constexpr uint32_t gptoss_ogs_block_n_small = 64;
constexpr uint32_t gptoss_ogs_threads       = 128;
constexpr uint32_t gptoss_ogs_shared_memory = 16384;

constexpr uint32_t gptoss_decode_swa_gluon_shared_memory  = 2112;
constexpr uint32_t gptoss_decode_full_gluon_shared_memory = 11776;

}  // namespace

hipError_t gptoss_q8_qkv_launch(hipFunction_t  function,
                                __half *        output,
                                const __half *  activation,
                                const int8_t *  values,
                                const __half *  scales,
                                const float *   bias,
                                uint32_t        n_tokens,
                                hipStream_t     stream) {
    int32_t n_tokens_i32    = static_cast<int32_t>(n_tokens);
    void *  global_scratch  = nullptr;
    void *  profile_scratch = nullptr;
    void * kernel_params[] = {
        &output, &activation, &values, &scales, &bias, &n_tokens_i32, &global_scratch, &profile_scratch,
    };
    const uint32_t blocks =
        ((n_tokens + gptoss_q8_block_m - 1) / gptoss_q8_block_m) *
        ((gptoss_qkv_size + gptoss_q8_block_n - 1) / gptoss_q8_block_n);
    return hipModuleLaunchKernel(
        function, blocks, 1, 1, gptoss_q8_threads, 1, 1, gptoss_q8_shared_memory, stream, kernel_params, nullptr);
}

hipError_t gptoss_q8_attention_output_launch(hipFunction_t  function,
                                             float *        output,
                                             const __half * activation,
                                             const int8_t * values,
                                             const __half * scales,
                                             const float *  bias,
                                             const float *  residual,
                                             uint32_t       n_tokens,
                                             hipStream_t    stream) {
    int32_t n_tokens_i32    = static_cast<int32_t>(n_tokens);
    void *  global_scratch  = nullptr;
    void *  profile_scratch = nullptr;
    void * kernel_params[] = {
        &output, &activation, &values, &scales, &bias, &residual, &n_tokens_i32, &global_scratch, &profile_scratch,
    };
    const uint32_t blocks =
        ((n_tokens + gptoss_q8_block_m - 1) / gptoss_q8_block_m) *
        ((gptoss_hidden_size + gptoss_q8_block_n - 1) / gptoss_q8_block_n);
    return hipModuleLaunchKernel(
        function, blocks, 1, 1, gptoss_q8_threads, 1, 1, gptoss_q8_shared_memory, stream, kernel_params, nullptr);
}

hipError_t gptoss_router_launch(hipFunction_t  function,
                                float *        output,
                                const float *  activation,
                                const float *  weight,
                                uint32_t       n_tokens,
                                hipStream_t    stream) {
    int32_t n_tokens_i32    = static_cast<int32_t>(n_tokens);
    void *  global_scratch  = nullptr;
    void *  profile_scratch = nullptr;
    void * kernel_params[] = {
        &output, &activation, &weight, &n_tokens_i32, &global_scratch, &profile_scratch,
    };
    const uint32_t blocks = (n_tokens + gptoss_router_block_m - 1) / gptoss_router_block_m;
    return hipModuleLaunchKernel(function, blocks, 1, 1, gptoss_router_threads, 1, 1, gptoss_router_shared_memory,
                                 stream, kernel_params, nullptr);
}

hipError_t gptoss_fa_launch(hipFunction_t  function,
                            __half *        output,
                            const __half *  query,
                            const __half *  key_cache,
                            const __half *  value_cache,
                            const float *   sinks,
                            const int32_t * block_table,
                            const int32_t * seq_lens,
                            int64_t         block_table_stride,
                            const int32_t * cu_seqlens_q,
                            uint32_t        n_sequences,
                            uint32_t        n_tokens,
                            hipStream_t     stream) {
    int32_t n_sequences_i32 = static_cast<int32_t>(n_sequences);
    void *  global_scratch  = nullptr;
    void *  profile_scratch = nullptr;
    void * kernel_params[] = {
        &output,          &query,           &key_cache,       &value_cache,
        &sinks,           &block_table,     &seq_lens,        &block_table_stride,
        &cu_seqlens_q,    &n_sequences_i32, &global_scratch,  &profile_scratch,
    };
    const uint32_t query_blocks = n_tokens / gptoss_fa_block_q + n_sequences;

    return hipModuleLaunchKernel(
        function, gptoss_kv_head_count, query_blocks, 1, gptoss_fa_threads, 1, 1, gptoss_fa_shared_memory, stream,
        kernel_params, nullptr);
}

hipError_t gptoss_ogs_w13_launch(hipFunction_t  function,
                                 __half *        output,
                                 const __half *  activation,
                                 const uint8_t * values,
                                 const uint8_t * scales,
                                 const float *   bias,
                                 const int32_t * gather_indices,
                                 const int32_t * expert_counts,
                                 const int32_t * route_offsets,
                                 const int32_t * block_offsets,
                                 const int32_t * block_schedule,
                                 uint32_t        schedule_capacity,
                                 bool            use_small_tiles,
                                 hipStream_t     stream) {
    int32_t grid_m_i32      = static_cast<int32_t>(schedule_capacity);
    void *  global_scratch  = nullptr;
    void *  profile_scratch = nullptr;
    void * kernel_params[] = {
        &output,          &output,           &activation,      &activation,
        &values,          &values,           &scales,          &bias,
        &gather_indices,  &expert_counts,    &route_offsets,   &block_offsets,
        &block_schedule,  &grid_m_i32,       &global_scratch,  &profile_scratch,
    };
    const uint32_t block_n = use_small_tiles ? gptoss_ogs_block_n_small : gptoss_ogs_block_n;
    const uint32_t grid_n  = (2 * gptoss_intermediate_size + block_n - 1) / block_n;
    return hipModuleLaunchKernel(function, schedule_capacity * grid_n, 1, 1, gptoss_ogs_threads, 1, 1,
                                 gptoss_ogs_shared_memory, stream, kernel_params, nullptr);
}

hipError_t gptoss_ogs_w2_launch(hipFunction_t  function,
                                float *         output,
                                const __half *  activation,
                                const uint8_t * values,
                                const uint8_t * scales,
                                const float *   bias,
                                const int32_t * scatter_indices,
                                uint32_t        route_count,
                                const int32_t * expert_counts,
                                const int32_t * route_offsets,
                                const int32_t * block_offsets,
                                const int32_t * block_schedule,
                                uint32_t        schedule_capacity,
                                bool            use_small_tiles,
                                hipStream_t     stream) {
    int32_t route_count_i32 = static_cast<int32_t>(route_count);
    int32_t grid_m_i32      = static_cast<int32_t>(schedule_capacity);
    void *  global_scratch  = nullptr;
    void *  profile_scratch = nullptr;
    void * kernel_params[] = {
        &output,           &output,          &activation,      &activation,      &values,         &values,
        &scales,           &bias,            &scatter_indices, &route_count_i32, &expert_counts,  &route_offsets,
        &block_offsets,    &block_schedule,  &grid_m_i32,      &global_scratch,  &profile_scratch,
    };
    const uint32_t block_n = use_small_tiles ? gptoss_ogs_block_n_small : gptoss_ogs_block_n;
    const uint32_t grid_n  = (gptoss_hidden_size + block_n - 1) / block_n;
    return hipModuleLaunchKernel(function, schedule_capacity * grid_n, 1, 1, gptoss_ogs_threads, 1, 1,
                                 gptoss_ogs_shared_memory, stream, kernel_params, nullptr);
}

hipError_t gptoss_decode_layer_gluon_launch(hipFunction_t                      function,
                                            bool                               swa,
                                            const gptoss_decode_layer_params & params,
                                            hipStream_t                        stream) {
    gptoss_decode_layer_params args = params;
    void *                     global_scratch  = nullptr;
    void *                     profile_scratch = nullptr;
    void * kernel_params[] = {
        &args.next,
        &args.cur,
        &args.rms_partials,
        &args.activation_scratch,
        &args.query,
        &args.attn_parts,
        &args.attn_meta,
        &args.router,
        &args.expert_ids,
        &args.expert_weights,
        &args.cache_k,
        &args.cache_v,
        &args.kv_rows,
        &args.attn_norm,
        &args.qkv_values,
        &args.attn_q_bias,
        &args.attn_k_bias,
        &args.attn_v_bias,
        &args.attn_output_values,
        &args.attn_output_bias,
        &args.attn_sinks,
        &args.post_attention_norm,
        &args.router_weight,
        &args.router_bias,
        &args.moe_down_values,
        &args.moe_gate_up_values,
        &args.moe_down_bias,
        &args.moe_gate_up_bias,
        &args.n_kv,
        &args.kv_write_row,
        &args.attn_parallel_blocks,
        &args.position,
        &args.rms_epsilon,
        &args.rope_freq_scale,
        &args.rope_ext_factor,
        &args.rope_attn_factor,
        &args.rope_corr_low,
        &args.rope_corr_high,
        &args.rope_theta_scale,
        &args.reuse_attention_rms,
        &global_scratch,
        &profile_scratch,
    };
    const uint32_t shared_memory =
        swa ? gptoss_decode_swa_gluon_shared_memory : gptoss_decode_full_gluon_shared_memory;
    return hipModuleLaunchCooperativeKernel(function, gptoss_decode_grid_blocks, 1, 1,
                                            gptoss_decode_block_x * gptoss_decode_block_y, 1, 1, shared_memory,
                                            stream, kernel_params);
}
