#include "gptoss-extension.h"

#include "extensions/llama-execution-extension.h"
#include "ggml.h"
#include "gptoss-repack-hip.h"
#include "llama-model.h"

#include <hip/hip_runtime_api.h>

#include <cstring>

static constexpr uint32_t gptoss_layer_count       = 24;
static constexpr uint32_t gptoss_hidden_size       = 2880;
static constexpr uint32_t gptoss_intermediate_size = 2880;
static constexpr uint32_t gptoss_expert_count      = 32;
static constexpr uint32_t gptoss_expert_used_count = 4;

static bool gptoss_model_init(llama_model * model) {
    const llama_hparams & hparams = model->hparams;

    if (hparams.n_layer_all != gptoss_layer_count || hparams.n_embd != gptoss_hidden_size ||
        hparams.n_ff_exp != gptoss_intermediate_size || hparams.n_expert != gptoss_expert_count ||
        hparams.n_expert_used != gptoss_expert_used_count) {
        return false;
    }

    const size_t scratch_size = 3 * ggml_nbytes(model->layers[0].ffn_gate_exps);
    uint8_t *    scratch      = nullptr;

    if (hipMalloc(&scratch, scratch_size) != hipSuccess) {
        return false;
    }

    bool success = true;

    for (uint32_t il = 0; il < gptoss_layer_count; ++il) {
        llama_layer & layer = model->layers[il];

        if (!gptoss_repack_qkv_launch(static_cast<uint8_t *>(layer.wq->data), static_cast<uint8_t *>(layer.wk->data),
                                      static_cast<uint8_t *>(layer.wv->data), scratch) ||
            !gptoss_repack_attention_output_launch(static_cast<uint8_t *>(layer.wo->data), scratch) ||
            !gptoss_repack_moe_launch(
                static_cast<uint8_t *>(layer.ffn_gate_exps->data), static_cast<uint8_t *>(layer.ffn_down_exps->data),
                static_cast<uint8_t *>(layer.ffn_up_exps->data), static_cast<float *>(layer.ffn_gate_exps_b->data),
                static_cast<float *>(layer.ffn_down_exps_b->data), static_cast<float *>(layer.ffn_up_exps_b->data),
                scratch)) {
            success = false;
            break;
        }
    }

    const bool synchronized = hipStreamSynchronize(nullptr) == hipSuccess;
    const bool freed        = hipFree(scratch) == hipSuccess;

    return success && synchronized && freed;
}

static bool gptoss_context_init(llama_context *) {
    return false;
}

static void gptoss_context_free(llama_context *) {}

static int gptoss_prefill(llama_context *, const llama_ubatch &, float *) {
    return -1;
}

static int gptoss_decode(llama_context *, const llama_ubatch &, float *) {
    return -1;
}

static size_t gptoss_tensor_alloc_size(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_MXFP4 || std::strstr(tensor->name, ".ffn_up_exps.weight") == nullptr) {
        return 0;
    }

    constexpr size_t ogs_alignment = 256;

    const size_t hidden       = static_cast<size_t>(tensor->ne[0]);
    const size_t intermediate = static_cast<size_t>(tensor->ne[1]);
    const size_t n_experts    = static_cast<size_t>(tensor->ne[2]);

    const size_t padded_hidden       = GGML_PAD(hidden, ogs_alignment);
    const size_t padded_intermediate = GGML_PAD(intermediate, ogs_alignment);

    const size_t packed_gate_size = n_experts * padded_intermediate * ggml_row_size(tensor->type, padded_hidden);
    const size_t packed_up_size   = packed_gate_size;
    const size_t packed_down_size = n_experts * padded_hidden * ggml_row_size(tensor->type, padded_intermediate);

    const size_t native_gate_size = ggml_nbytes(tensor);
    const size_t native_down_size = ggml_nbytes(tensor);

    return packed_gate_size + packed_up_size + packed_down_size - native_gate_size - native_down_size;
}

const llama_execution_extension gptoss_execution_extension = {
    gptoss_model_init, gptoss_context_init, gptoss_context_free,
    gptoss_prefill,    gptoss_decode,       gptoss_tensor_alloc_size,
};
