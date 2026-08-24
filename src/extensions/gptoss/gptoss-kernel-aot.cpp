#include "gptoss-kernel-aot.h"

#include "gptoss-config.h"

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
constexpr uint32_t gptoss_ogs_threads       = 128;
constexpr uint32_t gptoss_ogs_shared_memory = 16384;

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
    const uint32_t grid_n = (2 * gptoss_intermediate_size + gptoss_ogs_block_n - 1) / gptoss_ogs_block_n;
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
    const uint32_t grid_n = (gptoss_hidden_size + gptoss_ogs_block_n - 1) / gptoss_ogs_block_n;
    return hipModuleLaunchKernel(function, schedule_capacity * grid_n, 1, 1, gptoss_ogs_threads, 1, 1,
                                 gptoss_ogs_shared_memory, stream, kernel_params, nullptr);
}
