#include "llama-execution-extension.h"

#include "gptoss/gptoss-extension.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>

using llama_execution_extension_registry =
    std::array<const llama_execution_extension *, static_cast<size_t>(LLM_ARCH_UNKNOWN)>;

static llama_execution_extension_registry & get_execution_extension_registry() {
    static llama_execution_extension_registry registry = {};
    return registry;
}

bool llama_execution_extensions_register() {
    const char * value = std::getenv("LLAMA_EXECUTION_EXTENSION");

    if (value == nullptr || std::strcmp(value, "1") != 0) {
        return true;
    }

    return llama_execution_extension_register(LLM_ARCH_OPENAI_MOE, &gptoss_execution_extension);
}

bool llama_execution_extension_register(llm_arch arch, const llama_execution_extension * extension) {
    if (extension == nullptr || extension->model_init == nullptr || extension->context_init == nullptr ||
        extension->context_free == nullptr || extension->prefill == nullptr || extension->decode == nullptr) {
        return false;
    }

    auto &       registry = get_execution_extension_registry();
    const size_t index    = static_cast<size_t>(arch);

    if (index >= registry.size() || registry[index] != nullptr) {
        return false;
    }

    registry[index] = extension;
    return true;
}

const llama_execution_extension * llama_execution_extension_get(llm_arch arch) {
    const auto & registry = get_execution_extension_registry();
    const size_t index    = static_cast<size_t>(arch);

    return index < registry.size() ? registry[index] : nullptr;
}
