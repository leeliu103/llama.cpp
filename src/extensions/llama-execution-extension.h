#pragma once

#include "llama-arch.h"

#include "ggml.h"

#include <cstdint>

struct llama_context;
struct llama_memory_context_i;
struct llama_model;
struct llama_ubatch;
struct llm_graph_result;

struct llama_execution_extension_result {
    // false permits stock fallback only before memory apply or compute submission.
    bool handled = false;
    llm_graph_result * graph = nullptr;
    ggml_status status = GGML_STATUS_SUCCESS;
    uint64_t custom_stages = 0;
    bool metadata_only = false;
};

struct llama_execution_extension_stats {
    uint64_t eligible_tokens = 0;
    uint64_t eligible_no_output_tokens = 0;
    uint64_t custom_tokens = 0;
    uint64_t custom_stages = 0;
    uint64_t fallback_tokens = 0;
    uint64_t stock_prefill_tokens = 0;
    uint64_t failed_tokens = 0;
    uint64_t metadata_only_tokens = 0;
};

struct llama_execution_extension {
    const char * name;
    bool (*model_init)(llama_model * model);
    bool (*context_init)(llama_context * ctx);
    void (*context_free)(llama_context * ctx);

    // Must not apply memory, write KV values, or submit compute/collectives.
    bool (*supports)(const llama_context * ctx,
                     const llama_ubatch & ubatch,
                     const llama_memory_context_i * mctx);

    // A handled result uses the normal graph output and error paths in decode().
    llama_execution_extension_result (*execute)(llama_context * ctx,
                                                 const llama_ubatch & ubatch,
                                                 llama_memory_context_i * mctx);
};

const llama_execution_extension * llama_execution_extension_get(llm_arch arch);
