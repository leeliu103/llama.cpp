#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint32_t gptoss_hidden_size       = 2880;
constexpr uint32_t gptoss_intermediate_size = 2880;

static_assert(gptoss_hidden_size == gptoss_intermediate_size);

constexpr uint32_t gptoss_head_size        = 64;
constexpr uint32_t gptoss_query_head_count = 64;
constexpr uint32_t gptoss_kv_head_count    = 8;
constexpr uint32_t gptoss_query_size       = gptoss_query_head_count * gptoss_head_size;
constexpr uint32_t gptoss_key_value_size   = gptoss_kv_head_count * gptoss_head_size;
constexpr uint32_t gptoss_qkv_size         = gptoss_query_size + 2 * gptoss_key_value_size;

constexpr uint32_t gptoss_expert_used_count = 4;
constexpr uint32_t gptoss_vocabulary_size    = 201088;
constexpr uint32_t gptoss_swa_size           = 128;

struct gptoss_model_config {
    uint32_t layer_count;
    uint32_t expert_count;
};

constexpr gptoss_model_config gptoss_config_20b  = { 24,  32 };
constexpr gptoss_model_config gptoss_config_120b = { 36, 128 };

constexpr uint32_t gptoss_max_expert_count = gptoss_config_120b.expert_count;

constexpr uint32_t gptoss_quant_block_size = 32;
constexpr uint32_t gptoss_mxfp4_block_size = 32;
constexpr uint32_t gptoss_ogs_alignment     = 256;
constexpr uint32_t gptoss_ogs_block_m       = 64;
constexpr uint32_t gptoss_ogs_block_m_small = 16;
constexpr uint32_t gptoss_ogs_small_max_m   = 512;

constexpr uint32_t gptoss_mxfp4_padded_size =
    (gptoss_intermediate_size + gptoss_ogs_alignment - 1) / gptoss_ogs_alignment * gptoss_ogs_alignment;

constexpr size_t gptoss_qkv_values_size = static_cast<size_t>(gptoss_qkv_size) * gptoss_hidden_size;
constexpr size_t gptoss_attention_output_values_size =
    static_cast<size_t>(gptoss_hidden_size) * gptoss_query_size;

constexpr size_t gptoss_moe_down_values_size(uint32_t expert_count) {
    return static_cast<size_t>(expert_count) * gptoss_mxfp4_padded_size * gptoss_mxfp4_padded_size / 2;
}

constexpr size_t gptoss_moe_down_scales_size(uint32_t expert_count) {
    return static_cast<size_t>(expert_count) * gptoss_mxfp4_padded_size * gptoss_mxfp4_padded_size /
           gptoss_mxfp4_block_size;
}

constexpr size_t gptoss_moe_down_scales_offset(uint32_t expert_count) {
    return gptoss_moe_down_values_size(expert_count);
}

constexpr size_t gptoss_moe_gate_up_values_size(uint32_t expert_count) {
    return 2 * gptoss_moe_down_values_size(expert_count);
}

constexpr size_t gptoss_moe_gate_up_values_offset(uint32_t expert_count) {
    return gptoss_moe_down_values_size(expert_count) + gptoss_moe_down_scales_size(expert_count);
}

constexpr size_t gptoss_moe_gate_up_scales_offset(uint32_t expert_count) {
    return gptoss_moe_gate_up_values_offset(expert_count) + gptoss_moe_gate_up_values_size(expert_count);
}
