#pragma once

// The tensor-split decode kernel for Qwen3-235B-A22B under -sm tensor over 3 RPC devices: each RPC server runs one
// persistent kernel on its slices of the model's weights and KV cache (llama.cpp's own), the servers summing
// their partial results over their own link (rdna-megakernels, kernels/standalone/qwen3moe_shard.hip). Prompt
// processing stays llama.cpp's; single-token decode steps go to the kernel. LLAMA_MEGAKERNEL=1 enables it.

#include "ggml-backend.h"

#include <cstdint>
#include <memory>

struct llama_model;
struct llama_cparams;
struct llama_ubatch;
struct llama_memory_i;
struct llama_memory_context_i;

struct llama_megakernel_tp;

struct llama_megakernel_tp_deleter {
    void operator()(llama_megakernel_tp * mk) const;
};

using llama_megakernel_tp_ptr = std::unique_ptr<llama_megakernel_tp, llama_megakernel_tp_deleter>;

enum llama_megakernel_tp_result {
    LLAMA_MEGAKERNEL_TP_SKIPPED, // nothing done: decode with the graph
    LLAMA_MEGAKERNEL_TP_APPLIED, // the memory context was applied but the kernel failed: graph without applying it
    LLAMA_MEGAKERNEL_TP_DONE,    // decoded, logits written
};

// nullptr (with the reason in the log) when not enabled or not supported
llama_megakernel_tp_ptr llama_megakernel_tp_init(const llama_model & model, const llama_cparams & cparams,
                                                 ggml_backend_t meta_backend, llama_memory_i * memory);

// logits_out: n_vocab floats, or nullptr
llama_megakernel_tp_result llama_megakernel_tp_decode(llama_megakernel_tp * mk, const llama_ubatch & ubatch,
                                                      llama_memory_context_i * mctx, ggml_backend_sched_t sched,
                                                      float * logits_out);
