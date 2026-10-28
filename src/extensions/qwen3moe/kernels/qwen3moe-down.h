#pragma once

#include "ggml.h"
#include <hip/hip_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One token's selected expert activations. Strides are bytes.
// Writes only the local weighted MoE sum; does not add a residual.
struct qwen3moe_down_args {
    const void * down;
    const float * input;
    const int32_t * selected_experts;
    const float * routing_weights;
    float * output;
    enum ggml_type weight_type;
    int32_t input_size;
    int32_t output_rows;
    int32_t expert_count;
    int32_t selected_count;
    uint64_t down_row_stride;
    uint64_t down_expert_stride;
    uint64_t input_expert_stride;
    uint64_t routing_weight_stride;
    int32_t rows_per_block;
};

int qwen3moe_down_supported(const struct qwen3moe_down_args * args);

// Wave32 only. Enqueues one finite kernel without allocation or synchronization.
hipError_t qwen3moe_down_launch(const struct qwen3moe_down_args * args, hipStream_t stream);

#ifdef __cplusplus
}
#endif
