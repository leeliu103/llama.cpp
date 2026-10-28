#pragma once
#include "ggml.h"
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { QWEN3MOE_FFN_ROUTER_Q8_128_DOWN_ROWS = 32, QWEN3MOE_FFN_ROUTER_Q8_128_DOWN_MAPPING = 6 };
// Isolated approximate FFN: RMS/gamma, F32 router, deterministic top8 logit
// selection and selected-only softmax, then native Q8-128 expert projections.
// Fixed Qwen3-235B TP3 shapes. No residual or communication is included.
// Every pointer is device storage. All writable spans must be disjoint from
// each other and all reads. Logits cannot alias H: CTAs read them asynchronously.
struct qwen3moe_ffn_router_q8_128_args {
    const void * gate;
    const void * up;
    const void * down;
    const float * input;
    const float * ffn_norm;
    const float * router;
    float * output;                  // 4096 floats
    float * activation_scratch;      // 8 x 512 floats
    float * router_logits;           // 128 floats
    int32_t * selected_experts_out;  // 8 IDs; written only by CTA0 after routing
    float * routing_weights_out;     // 8 floats; written only by CTA0 after routing
    uint32_t * arrival_counter;      // H phase
    uint32_t * router_arrival_counter;
    uint32_t * sticky_error;
    int down_type;
    uint64_t gate_row_stride;
    uint64_t gate_expert_stride;
    uint64_t up_row_stride;
    uint64_t up_expert_stride;
    uint64_t down_row_stride;
    uint64_t down_expert_stride;
    uint64_t router_row_stride;
    float rms_epsilon;
    int grid_blocks;                 // fixed32
    int stream_depth;                // fixed4
};
enum qwen3moe_ffn_router_q8_128_error {
    QWEN3MOE_FFN_ROUTER_Q8_128_INVALID_EXPERT = 1,
    QWEN3MOE_FFN_ROUTER_Q8_128_BARRIER_TIMEOUT = 2,
    QWEN3MOE_FFN_ROUTER_Q8_128_BARRIER_COUNT = 4,
    QWEN3MOE_FFN_ROUTER_Q8_128_NONFINITE = 8,
};
struct qwen3moe_ffn_router_q8_128_launch_info {
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
int qwen3moe_ffn_router_q8_128_supported(const struct qwen3moe_ffn_router_q8_128_args * args);
hipError_t qwen3moe_ffn_router_q8_128_query(const struct qwen3moe_ffn_router_q8_128_args * args,
                                         struct qwen3moe_ffn_router_q8_128_launch_info * info);
// Query outside capture/timing. Enqueue resets both independent counters with
// separate memsets, then submits one finite cooperative kernel. It never resets
// sticky_error. Caller initializes sticky_error and synchronizes before reuse.
// Preseeded errors return after router barrier: router arrivals32, H arrivals0,
// H/output/logits/ID/weight outputs untouched. Later-phase failure may leave
// intermediate trace outputs changed; the entire sequence is invalid.
hipError_t qwen3moe_ffn_router_q8_128_enqueue(const struct qwen3moe_ffn_router_q8_128_args * args,
                                           const struct qwen3moe_ffn_router_q8_128_launch_info * info, hipStream_t stream);
hipError_t qwen3moe_ffn_router_q8_128_launch(const struct qwen3moe_ffn_router_q8_128_args * args, hipStream_t stream);
#ifdef __cplusplus
}
#endif
