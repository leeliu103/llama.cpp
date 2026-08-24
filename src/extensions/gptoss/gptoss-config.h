#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint32_t gptoss_layer_count       = 24;
constexpr uint32_t gptoss_hidden_size       = 2880;
constexpr uint32_t gptoss_intermediate_size = 2880;

static_assert(gptoss_hidden_size == gptoss_intermediate_size);

constexpr uint32_t gptoss_head_size        = 64;
constexpr uint32_t gptoss_query_head_count = 64;
constexpr uint32_t gptoss_kv_head_count    = 8;
constexpr uint32_t gptoss_query_size       = gptoss_query_head_count * gptoss_head_size;
constexpr uint32_t gptoss_key_value_size   = gptoss_kv_head_count * gptoss_head_size;
constexpr uint32_t gptoss_qkv_size         = gptoss_query_size + 2 * gptoss_key_value_size;

constexpr uint32_t gptoss_expert_count       = 32;
constexpr uint32_t gptoss_expert_used_count = 4;
constexpr uint32_t gptoss_vocabulary_size    = 201088;
constexpr uint32_t gptoss_swa_size           = 128;

constexpr uint32_t gptoss_quant_block_size = 32;
constexpr uint32_t gptoss_mxfp4_block_size = 32;
constexpr uint32_t gptoss_ogs_alignment     = 256;
constexpr uint32_t gptoss_ogs_block_m       = 64;

constexpr uint32_t gptoss_mxfp4_padded_size =
    (gptoss_intermediate_size + gptoss_ogs_alignment - 1) / gptoss_ogs_alignment * gptoss_ogs_alignment;

constexpr size_t gptoss_qkv_values_size = static_cast<size_t>(gptoss_qkv_size) * gptoss_hidden_size;
constexpr size_t gptoss_attention_output_values_size =
    static_cast<size_t>(gptoss_hidden_size) * gptoss_query_size;

constexpr size_t gptoss_moe_down_values_size =
    static_cast<size_t>(gptoss_expert_count) * gptoss_mxfp4_padded_size * gptoss_mxfp4_padded_size / 2;
constexpr size_t gptoss_moe_down_scales_size =
    static_cast<size_t>(gptoss_expert_count) * gptoss_mxfp4_padded_size * gptoss_mxfp4_padded_size /
    gptoss_mxfp4_block_size;
constexpr size_t gptoss_moe_down_scales_offset    = gptoss_moe_down_values_size;
constexpr size_t gptoss_moe_gate_up_values_size   = 2 * gptoss_moe_down_values_size;
constexpr size_t gptoss_moe_gate_up_values_offset = gptoss_moe_down_values_size + gptoss_moe_down_scales_size;
constexpr size_t gptoss_moe_gate_up_scales_offset =
    gptoss_moe_gate_up_values_offset + gptoss_moe_gate_up_values_size;
