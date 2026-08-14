#include "gptoss-extension.h"

#include "extensions/llama-execution-extension.h"
#include "ggml.h"
#include "gptoss-repack-hip.h"
#include "llama-context.h"
#include "llama-model.h"

#include <hip/hip_runtime_api.h>

#include <cstring>

#define GPTOSS_AOT_DIR "/app/llama.cpp/src/extensions/gptoss/build/gfx1201"

static constexpr uint32_t gptoss_layer_count       = 24;
static constexpr uint32_t gptoss_hidden_size       = 2880;
static constexpr uint32_t gptoss_intermediate_size = 2880;
static constexpr uint32_t gptoss_expert_count      = 32;
static constexpr uint32_t gptoss_expert_used_count = 4;

struct gptoss_context_state {
    hipModule_t q8_qkv_module      = nullptr;
    hipModule_t q8_attn_out_module = nullptr;
    hipModule_t fa_full_module     = nullptr;
    hipModule_t fa_sw128_module    = nullptr;
    hipModule_t ogs_w13_module     = nullptr;
    hipModule_t ogs_w2_module      = nullptr;
};

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

static void gptoss_context_free(llama_context * ctx) {
    auto * state = static_cast<gptoss_context_state *>(ctx->execution_extension_state);

    if (state == nullptr) {
        return;
    }

    if (state->q8_qkv_module != nullptr) {
        (void) hipModuleUnload(state->q8_qkv_module);
    }
    if (state->q8_attn_out_module != nullptr) {
        (void) hipModuleUnload(state->q8_attn_out_module);
    }
    if (state->fa_full_module != nullptr) {
        (void) hipModuleUnload(state->fa_full_module);
    }
    if (state->fa_sw128_module != nullptr) {
        (void) hipModuleUnload(state->fa_sw128_module);
    }
    if (state->ogs_w13_module != nullptr) {
        (void) hipModuleUnload(state->ogs_w13_module);
    }
    if (state->ogs_w2_module != nullptr) {
        (void) hipModuleUnload(state->ogs_w2_module);
    }

    delete state;
    ctx->execution_extension_state = nullptr;
}

static bool gptoss_context_init(llama_context * ctx) {
    auto * state                   = new gptoss_context_state;
    ctx->execution_extension_state = state;

    if (hipModuleLoad(&state->q8_qkv_module, GPTOSS_AOT_DIR "/q8_qkv.hsaco") != hipSuccess ||
        hipModuleLoad(&state->q8_attn_out_module, GPTOSS_AOT_DIR "/q8_attn_out.hsaco") != hipSuccess ||
        hipModuleLoad(&state->fa_full_module, GPTOSS_AOT_DIR "/fa_full.hsaco") != hipSuccess ||
        hipModuleLoad(&state->fa_sw128_module, GPTOSS_AOT_DIR "/fa_sw128.hsaco") != hipSuccess ||
        hipModuleLoad(&state->ogs_w13_module, GPTOSS_AOT_DIR "/ogs_w13.hsaco") != hipSuccess ||
        hipModuleLoad(&state->ogs_w2_module, GPTOSS_AOT_DIR "/ogs_w2.hsaco") != hipSuccess) {
        gptoss_context_free(ctx);
        return false;
    }

    return true;
}

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
