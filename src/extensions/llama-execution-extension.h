#pragma once

#include "llama-arch.h"

#include <cstddef>

struct ggml_tensor;
struct llama_context;
struct llama_model;
struct llama_ubatch;

struct llama_execution_extension {
    bool (*model_init)(llama_model * model);
    bool (*context_init)(llama_context * ctx);
    void (*context_free)(llama_context * ctx);
    int (*prefill)(llama_context * ctx, const llama_ubatch & ubatch, float * logits_out);
    int (*decode)(llama_context * ctx, const llama_ubatch & ubatch, float * logits_out);
    size_t (*tensor_alloc_size)(const ggml_tensor * tensor);
};

bool llama_execution_extension_register(llm_arch arch, const llama_execution_extension * extension);

const llama_execution_extension * llama_execution_extension_get(llm_arch arch);
