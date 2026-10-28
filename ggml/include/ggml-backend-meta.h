#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of simple backends represented by a meta backend.
GGML_API size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend);

// Select a plan using cached capabilities. This does not submit compute or collectives.
GGML_API bool ggml_backend_meta_execution_extension_begin(ggml_backend_t backend, const char * plan_name);

// Clear the selection and report successful custom stages across all ranks.
// A false return indicates an execution failure; the token must not be retried through stock.
GGML_API bool ggml_backend_meta_execution_extension_end(ggml_backend_t backend, uint64_t * custom_stages);

#ifdef __cplusplus
}
#endif
