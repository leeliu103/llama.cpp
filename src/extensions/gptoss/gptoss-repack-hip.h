#pragma once

#include <cstdint>

bool gptoss_repack_qkv_launch(uint8_t * q, uint8_t * k, uint8_t * v, uint8_t * scratch);

bool gptoss_repack_attention_output_launch(uint8_t * weight, uint8_t * scratch);

bool gptoss_repack_moe_launch(uint8_t * gate,
                              uint8_t * down,
                              uint8_t * up,
                              float *   gate_bias,
                              float *   down_bias,
                              float *   up_bias,
                              uint32_t  expert_count,
                              uint8_t * scratch);
