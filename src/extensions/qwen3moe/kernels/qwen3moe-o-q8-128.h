#pragma once

#include "ggml-backend.h"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// Internal decode O projection. The backend owns graph admission and the
// once-per-token sticky status reset/seal.
enum qwen3moe_o_q8_128_error : uint32_t {
    QWEN3MOE_O_Q8_128_NONFINITE = 8,
};

struct qwen3moe_o_q8_128_args {
    const void * weights;
    const float * input;
    float * output;
    uint32_t * status;
    size_t weight_bytes, input_bytes, output_bytes;
    uint64_t row_stride;
    int columns;
};

// Checks metadata/ranges only. Real allocation and graph bindings are checked
// by the backend. All four byte ranges must be disjoint.
bool qwen3moe_o_q8_128_supported(const qwen3moe_o_q8_128_args * args);
// Does not reset status. A prior error or nonfinite input leaves zero output;
// nonfinite output rows are replaced with zero and mark the token failed.
// Uses only workgroup-local barriers, with no host synchronization.
hipError_t qwen3moe_o_q8_128_enqueue(const qwen3moe_o_q8_128_args * args, hipStream_t stream);
