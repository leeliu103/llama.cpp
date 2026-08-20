#include "gptoss-aot.h"

#include "gptoss-config.h"

#include <cstddef>

namespace {

constexpr uint32_t gptoss_q8_block_m       = 64;
constexpr uint32_t gptoss_q8_block_n       = 128;
constexpr uint32_t gptoss_q8_threads       = 128;
constexpr uint32_t gptoss_q8_shared_memory = 16384;

constexpr uint32_t gptoss_fa_block_q       = 8;
constexpr uint32_t gptoss_fa_threads       = 128;
constexpr uint32_t gptoss_fa_shared_memory = 8192;

constexpr uint32_t gptoss_ogs_block_n       = 128;
constexpr uint32_t gptoss_ogs_threads       = 128;
constexpr uint32_t gptoss_ogs_shared_memory = 16384;

static_assert(sizeof(gptoss_fa_params) == 96);
static_assert(sizeof(gptoss_ogs_w13_params) == 128);
static_assert(sizeof(gptoss_ogs_w2_params) == 136);

template <typename T>
hipError_t gptoss_aot_launch(hipFunction_t function,
                             uint32_t      blocks,
                             uint32_t      threads,
                             uint32_t      shared_memory,
                             T &           params,
                             hipStream_t   stream) {
    size_t params_size = sizeof(params);
    void * config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &params, HIP_LAUNCH_PARAM_BUFFER_SIZE, &params_size, HIP_LAUNCH_PARAM_END,
    };

    return hipModuleLaunchKernel(function, blocks, 1, 1, threads, 1, 1, shared_memory, stream, nullptr, config);
}

}  // namespace

hipError_t gptoss_q8_qkv_launch(hipFunction_t function, gptoss_q8_qkv_params & params, hipStream_t stream) {
    void * kernel_params[] = {
        &params.output, &params.activation, &params.values,         &params.scales,
        &params.bias,   &params.n_tokens,   &params.global_scratch, &params.profile_scratch,
    };
    const uint32_t blocks =
        ((static_cast<uint32_t>(params.n_tokens) + gptoss_q8_block_m - 1) / gptoss_q8_block_m) *
        ((gptoss_qkv_size + gptoss_q8_block_n - 1) / gptoss_q8_block_n);
    return hipModuleLaunchKernel(
        function, blocks, 1, 1, gptoss_q8_threads, 1, 1, gptoss_q8_shared_memory, stream, kernel_params, nullptr);
}

hipError_t gptoss_q8_attention_output_launch(hipFunction_t                     function,
                                             gptoss_q8_attention_output_params & params,
                                             hipStream_t                       stream) {
    void * kernel_params[] = {
        &params.output,   &params.activation, &params.values,         &params.scales,          &params.bias,
        &params.residual, &params.n_tokens,   &params.global_scratch, &params.profile_scratch,
    };
    const uint32_t blocks =
        ((static_cast<uint32_t>(params.n_tokens) + gptoss_q8_block_m - 1) / gptoss_q8_block_m) *
        ((gptoss_hidden_size + gptoss_q8_block_n - 1) / gptoss_q8_block_n);
    return hipModuleLaunchKernel(
        function, blocks, 1, 1, gptoss_q8_threads, 1, 1, gptoss_q8_shared_memory, stream, kernel_params, nullptr);
}

hipError_t gptoss_fa_launch(hipFunction_t function, uint32_t n_tokens, gptoss_fa_params & params, hipStream_t stream) {
    size_t params_size = sizeof(params);
    void * config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &params, HIP_LAUNCH_PARAM_BUFFER_SIZE, &params_size, HIP_LAUNCH_PARAM_END,
    };
    const uint32_t query_blocks = n_tokens / gptoss_fa_block_q + static_cast<uint32_t>(params.n_sequences);

    return hipModuleLaunchKernel(
        function, gptoss_kv_head_count, query_blocks, 1, gptoss_fa_threads, 1, 1, gptoss_fa_shared_memory, stream,
        nullptr, config);
}

hipError_t gptoss_ogs_w13_launch(hipFunction_t function, gptoss_ogs_w13_params & params, hipStream_t stream) {
    const uint32_t grid_n = (2 * gptoss_intermediate_size + gptoss_ogs_block_n - 1) / gptoss_ogs_block_n;
    return gptoss_aot_launch(function, static_cast<uint32_t>(params.grid_m) * grid_n, gptoss_ogs_threads,
                             gptoss_ogs_shared_memory, params, stream);
}

hipError_t gptoss_ogs_w2_launch(hipFunction_t function, gptoss_ogs_w2_params & params, hipStream_t stream) {
    const uint32_t grid_n = (gptoss_hidden_size + gptoss_ogs_block_n - 1) / gptoss_ogs_block_n;
    return gptoss_aot_launch(function, static_cast<uint32_t>(params.grid_m) * grid_n, gptoss_ogs_threads,
                             gptoss_ogs_shared_memory, params, stream);
}
