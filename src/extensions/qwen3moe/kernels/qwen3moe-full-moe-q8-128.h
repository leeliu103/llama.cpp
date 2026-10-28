#pragma once

#include "ggml.h"
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { QWEN3MOE_FULL_MOE_Q8_128_DOWN_ROWS = 32, QWEN3MOE_FULL_MOE_Q8_128_DOWN_MAPPING = 4 };

// Approximate native-weight streaming: one F32 scale per 128 activations.
// Experimental fixed Qwen3-235B TP3 region: K=4096, M=512, E=128,
// selected experts=8. Gate/up Q4_K; down Q4_K or Q6_K. No residual.
// All pointers are device pointers. Caller owns allocations and the stream.
// The output, activation scratch (4096 floats), arrival counter (one u32),
// and sticky error (one u32) must be mutually disjoint and disjoint from reads.
// Initialize sticky_error=0 before a sequence, and read it after the existing
// synchronization. Any nonzero error invalidates the entire sequence; never
// retry partially submitted work through stock. Scratch/state belong to one
// serialized stream and must not be shared by overlapping launches.
struct qwen3moe_full_moe_q8_128_args {
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
    int grid_blocks;         // launch 32 or 64; query/enqueue always reject 128
    int stream_depth;  // 2 or 4 native weight chunks in flight
};

enum qwen3moe_full_moe_q8_128_error {
    QWEN3MOE_FULL_MOE_Q8_128_INVALID_EXPERT = 1,
    QWEN3MOE_FULL_MOE_Q8_128_BARRIER_TIMEOUT = 2,
    QWEN3MOE_FULL_MOE_Q8_128_BARRIER_COUNT = 4,
};

struct qwen3moe_full_moe_q8_128_launch_info {
    int device;
    int multiprocessors;
    int active_blocks_per_multiprocessor;
    int grid_blocks;
    int down_type;
    int stream_depth;
    int down_rows;
    int down_mapping;
    uint64_t barrier_timeout_ticks;
};

// Pure descriptor validation. Does not call HIP or dereference device pointers.
int qwen3moe_full_moe_q8_128_supported(const struct qwen3moe_full_moe_q8_128_args * args);
// Queries current device and occupancy of the exact kernel specialization.
// Does not allocate, select a device, submit work, or synchronize.
hipError_t qwen3moe_full_moe_q8_128_query(const struct qwen3moe_full_moe_q8_128_args * args,
                                  struct qwen3moe_full_moe_q8_128_launch_info * info);
// Uses a successful query for this device/specialization; caller retains that
// device and serialized stream. No queries, allocation, or synchronization.
// Resets arrivals on every invocation, including captured replay.
hipError_t qwen3moe_full_moe_q8_128_enqueue(const struct qwen3moe_full_moe_q8_128_args * args,
                                    const struct qwen3moe_full_moe_q8_128_launch_info * info, hipStream_t stream);
// Queries residency, resets only the arrival counter on stream, and submits a
// finite cooperative kernel. Sticky error is deliberately not reset here.
hipError_t qwen3moe_full_moe_q8_128_launch(const struct qwen3moe_full_moe_q8_128_args * args, hipStream_t stream);

#ifdef __cplusplus
}
#endif
