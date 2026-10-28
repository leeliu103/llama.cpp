#include "llama-execution-extension.h"

#include "qwen3moe/qwen3moe-extension.h"

#include <cstdlib>
#include <cstring>

const llama_execution_extension * llama_execution_extension_get(llm_arch arch) {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_EXECUTION_EXTENSION");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();

    if (!enabled) {
        return nullptr;
    }

    switch (arch) {
        case LLM_ARCH_QWEN3MOE:
            return &qwen3moe_execution_extension;
        default:
            return nullptr;
    }
}
