#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

class llama_hip_workspace_cursor;

struct gptoss_prefill_buffers {
    int32_t * tokens;
    int32_t * positions;

    int64_t * base_write_rows;
    int64_t * swa_write_rows;
    int32_t * base_block_table;
    int32_t * swa_block_table;
    int32_t * cu_seqlens_q;
    int32_t * base_seq_lens;
    int32_t * swa_seq_lens;

    float *  cur;
    float *  next;
    __half * norm;
    float *  rope_base;
    float *  rope_swa;
    __half * qkv;
    __half * q;

    float *   router_logits;
    int32_t * selected_ids;
    float *   selected_weights;
    int32_t * gather_indices;
    int32_t * scatter_indices;
    int32_t * expert_counts;
    int32_t * route_offsets;
    int32_t * block_offsets;
    int32_t * block_schedule;
    __half *  expert_activations;
    float *   expert_outputs;
    uint32_t  route_count;
    uint32_t  schedule_capacity;

    float * logits;
};

gptoss_prefill_buffers gptoss_make_prefill_buffers(llama_hip_workspace_cursor & cursor,
                                                   uint32_t                     n_tokens,
                                                   uint32_t                     n_outputs,
                                                   uint32_t                     expert_count,
                                                   uint32_t                     n_sequences,
                                                   uint32_t                     n_base_table_elements,
                                                   uint32_t                     n_swa_table_elements);

struct gptoss_decode_buffers {
    int32_t * token;
    int32_t * base_rows;
    int32_t * swa_rows;

    float *   cur;
    float *   next;
    float *   rms_partials;
    __half *  activation_scratch;
    __half *  query;
    float *   router_scores;
    int32_t * selected_experts;
    float *   selected_weights;
    float *   attention_parts;
    float2 *  attention_meta;
    uint32_t  attention_partitions;

    float * logits;
};

gptoss_decode_buffers gptoss_make_decode_buffers(llama_hip_workspace_cursor & cursor,
                                                 size_t                       n_base_rows,
                                                 size_t                       n_swa_rows,
                                                 bool                         output,
                                                 uint32_t                     expert_count);
