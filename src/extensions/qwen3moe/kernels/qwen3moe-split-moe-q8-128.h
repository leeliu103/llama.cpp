#pragma once
#include "ggml.h"
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { QWEN3MOE_SPLIT_MOE_Q8_128_DOWN_ROWS = 32, QWEN3MOE_SPLIT_MOE_Q8_128_DOWN_MAPPING = 7 };
// Two ordinary kernels with native128 arithmetic. Stream order (and captured
// graph dependency) publishes complete H before down starts; no grid barrier.
// Pointers/layouts match the native fixture. arrival_counter is reserved for
// comparison and must remain untouched. Caller initializes/checks sticky_error
// outside the ordered sequence. Never retry after partial submission.
struct qwen3moe_split_moe_q8_128_args {
    const void * gate;
    const void * up;
    const void * down;
    const float * input;
    const int32_t * selected_experts;
    const float * routing_weights;
    float * output;
    float * activation_scratch;
    uint32_t * arrival_counter;
    uint32_t * sticky_error;
    int down_type;
    uint64_t gate_row_stride;
    uint64_t gate_expert_stride;
    uint64_t up_row_stride;
    uint64_t up_expert_stride;
    uint64_t down_row_stride;
    uint64_t down_expert_stride;
    uint64_t routing_stride;
    int grid_blocks; // 32, 64 or 128 for both independent phases
    int stream_depth; // 2 or 4
};
enum qwen3moe_split_moe_q8_128_error { QWEN3MOE_SPLIT_MOE_Q8_128_INVALID_EXPERT = 1 };
struct qwen3moe_split_moe_q8_128_launch_info {
    int device;
    int multiprocessors;
    int gate_up_active_blocks_per_multiprocessor;
    int down_active_blocks_per_multiprocessor;
    int grid_blocks;
    int down_type;
    int stream_depth;
    int down_rows;
    int down_mapping;
};
int qwen3moe_split_moe_q8_128_supported(const struct qwen3moe_split_moe_q8_128_args * args);
// Reports each phase's occupancy; no full-grid residency requirement exists.
hipError_t qwen3moe_split_moe_q8_128_query(const struct qwen3moe_split_moe_q8_128_args * args,
                                         struct qwen3moe_split_moe_q8_128_launch_info * info);
// Enqueue exactly two ordinary launches on one explicit serialized stream.
// No allocation, query, reset, host wait, cooperative launch or counter polling.
hipError_t qwen3moe_split_moe_q8_128_enqueue(const struct qwen3moe_split_moe_q8_128_args * args,
                                           const struct qwen3moe_split_moe_q8_128_launch_info * info, hipStream_t stream);
hipError_t qwen3moe_split_moe_q8_128_launch(const struct qwen3moe_split_moe_q8_128_args * args, hipStream_t stream);
#ifdef __cplusplus
}
#endif
