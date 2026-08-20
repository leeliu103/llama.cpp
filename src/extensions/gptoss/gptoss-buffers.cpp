#include "gptoss-buffers.h"

#include "extensions/hip-workspace.h"
#include "gptoss-config.h"
#include "gptoss-kernel.h"

#include <algorithm>

namespace {

constexpr uint32_t gptoss_max_attention_parts = 12;

}  // namespace

gptoss_prefill_buffers gptoss_make_prefill_buffers(llama_hip_workspace_cursor & cursor,
                                                   uint32_t                     n_tokens,
                                                   bool                         output,
                                                   uint32_t                     n_sequences,
                                                   uint32_t                     n_base_table_elements,
                                                   uint32_t                     n_swa_table_elements) {
    gptoss_prefill_buffers result = {};

    result.route_count       = n_tokens * gptoss_expert_used_count;
    result.schedule_capacity =
        (result.route_count + gptoss_ogs_block_m - 1) / gptoss_ogs_block_m + gptoss_expert_count - 1;

    result.tokens             = cursor.take<int32_t>(n_tokens);
    result.positions          = cursor.take<int32_t>(n_tokens);
    result.base_write_rows    = cursor.take<int64_t>(n_tokens);
    result.swa_write_rows     = cursor.take<int64_t>(n_tokens);
    result.base_block_table   = cursor.take<int32_t>(n_base_table_elements);
    result.swa_block_table    = cursor.take<int32_t>(n_swa_table_elements);
    result.cu_seqlens_q       = cursor.take<int32_t>(n_sequences + 1);
    result.base_seq_lens      = cursor.take<int32_t>(n_sequences);
    result.swa_seq_lens       = cursor.take<int32_t>(n_sequences);
    result.cur                = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.next               = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.norm               = cursor.take<__half>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.rope_base          = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_head_size);
    result.rope_swa           = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_head_size);
    result.qkv                = cursor.take<__half>(static_cast<size_t>(n_tokens) * gptoss_qkv_size);
    result.q                  = cursor.take<__half>(static_cast<size_t>(n_tokens) * gptoss_query_size);
    result.router_logits      = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_expert_count);
    result.selected_ids       = cursor.take<int32_t>(result.route_count);
    result.selected_weights   = cursor.take<float>(result.route_count);
    result.gather_indices     = cursor.take<int32_t>(result.route_count);
    result.scatter_indices    = cursor.take<int32_t>(result.route_count);
    result.expert_counts      = cursor.take<int32_t>(gptoss_expert_count);
    result.route_offsets      = cursor.take<int32_t>(gptoss_expert_count + 1);
    result.block_offsets      = cursor.take<int32_t>(gptoss_expert_count + 1);
    result.block_schedule     = cursor.take<int32_t>(result.schedule_capacity);
    result.expert_activations =
        cursor.take<__half>(static_cast<size_t>(result.route_count) * gptoss_intermediate_size);
    result.expert_outputs     =
        cursor.take<float>(static_cast<size_t>(result.route_count) * gptoss_hidden_size);
    result.logits             = cursor.take<float>(output ? gptoss_vocabulary_size : 0);

    return result;
}

gptoss_decode_buffers gptoss_make_decode_buffers(llama_hip_workspace_cursor & cursor,
                                                 size_t                       n_base_rows,
                                                 size_t                       n_swa_rows,
                                                 bool                         output) {
    const uint32_t partitions = static_cast<uint32_t>(std::min<size_t>(
        gptoss_max_attention_parts,
        std::max<size_t>(2, (n_base_rows + gptoss_swa_size - 1) / gptoss_swa_size)));

    gptoss_decode_buffers result = {};

    result.token            = cursor.take<int32_t>(1);
    result.base_rows        = cursor.take<int32_t>(n_base_rows + n_swa_rows);
    result.swa_rows         = result.base_rows == nullptr ? nullptr : result.base_rows + n_base_rows;
    result.cur              = cursor.take<float>(gptoss_hidden_size);
    result.next             = cursor.take<float>(gptoss_hidden_size);
    result.rms_partials     = cursor.take<float>(gptoss_decode_grid_blocks);
    result.activation_scratch =
        cursor.take<__half>(gptoss_hidden_size * (1 + gptoss_expert_used_count));
    result.query            = cursor.take<__half>(gptoss_query_size);
    result.router_scores    = cursor.take<float>(gptoss_expert_count);
    result.selected_experts = cursor.take<int32_t>(gptoss_expert_used_count);
    result.selected_weights = cursor.take<float>(gptoss_expert_used_count);
    result.attention_parts =
        cursor.take<float>(static_cast<size_t>(gptoss_query_head_count) * partitions * gptoss_head_size);
    result.attention_meta       = cursor.take<float2>(static_cast<size_t>(gptoss_query_head_count) * partitions);
    result.attention_partitions = partitions;
    result.logits               = cursor.take<float>(output ? gptoss_vocabulary_size : 0);

    return result;
}
