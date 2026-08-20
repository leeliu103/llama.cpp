#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

constexpr uint32_t gptoss_fa_tile_size = 32;

struct gptoss_q8_qkv_params {
    void *       output;
    const void * activation;
    const void * values;
    const void * scales;
    const void * bias;
    int32_t      n_tokens;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_q8_attention_output_params {
    void *       output;
    const void * activation;
    const void * values;
    const void * scales;
    const void * bias;
    const void * residual;
    int32_t      n_tokens;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_fa_params {
    void *       output;
    const void * query;
    const void * key_cache;
    const void * value_cache;
    const void * sinks;
    const void * block_table;
    const void * seq_lens;
    int64_t      block_table_stride;
    const void * cu_seqlens_q;
    int32_t      n_sequences;
    uint32_t     padding;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_ogs_w13_params {
    void *       output;
    void *       output_ptr;
    const void * activation;
    const void * activation_ptr;
    const void * values;
    const void * values_ptr;
    const void * scales;
    const void * bias;
    const void * gather_indices;
    const void * expert_counts;
    const void * route_offsets;
    const void * block_offsets;
    const void * block_schedule;
    int32_t      grid_m;
    uint32_t     padding;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_ogs_w2_params {
    void *       output;
    void *       output_ptr;
    const void * activation;
    const void * activation_ptr;
    const void * values;
    const void * values_ptr;
    const void * scales;
    const void * bias;
    const void * scatter_indices;
    int32_t      route_count;
    uint32_t     padding_0;
    const void * expert_counts;
    const void * route_offsets;
    const void * block_offsets;
    const void * block_schedule;
    int32_t      grid_m;
    uint32_t     padding_1;
    void *       global_scratch;
    void *       profile_scratch;
};

hipError_t gptoss_q8_qkv_launch(hipFunction_t function, gptoss_q8_qkv_params & params, hipStream_t stream);

hipError_t gptoss_q8_attention_output_launch(hipFunction_t                     function,
                                             gptoss_q8_attention_output_params & params,
                                             hipStream_t                       stream);

hipError_t gptoss_fa_launch(hipFunction_t function, uint32_t n_tokens, gptoss_fa_params & params, hipStream_t stream);

hipError_t gptoss_ogs_w13_launch(hipFunction_t function, gptoss_ogs_w13_params & params, hipStream_t stream);

hipError_t gptoss_ogs_w2_launch(hipFunction_t function, gptoss_ogs_w2_params & params, hipStream_t stream);
