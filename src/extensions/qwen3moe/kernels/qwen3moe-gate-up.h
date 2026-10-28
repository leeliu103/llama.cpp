#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One token, one shared input vector, and separate Q4_K gate/up matrices.
// All strides are bytes. Rows within an output expert slot are contiguous.
struct qwen3moe_gate_up_q4_k_args {
    const void * gate;
    const void * up;
    const float * input;
    const int32_t * selected_experts;
    float * output;
    int32_t input_size;
    int32_t output_rows;
    int32_t expert_count;
    int32_t selected_count;
    uint64_t gate_row_stride;
    uint64_t gate_expert_stride;
    uint64_t up_row_stride;
    uint64_t up_expert_stride;
    uint64_t output_expert_stride;
    int32_t rows_per_wave;
};

// Pure descriptor check. Device pointers and selected expert IDs must be valid.
int qwen3moe_gate_up_q4_k_supported(const struct qwen3moe_gate_up_q4_k_args * args);

// Requires a wave32 device. Enqueues one finite kernel on the current device.
// No allocations, device selection, synchronization, or global state changes.
hipError_t qwen3moe_gate_up_q4_k_launch(
        const struct qwen3moe_gate_up_q4_k_args * args, hipStream_t stream);

#ifdef __cplusplus
}
#endif
