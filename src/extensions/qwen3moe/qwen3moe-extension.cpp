#include "qwen3moe-extension.h"

#include "extensions/llama-execution-extension.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend-meta.h"

#include <memory>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct qwen3moe_context_state {
    bool supported_model = false;
    ggml_backend_t backend = nullptr;
    const char * plan_name = "qwen3moe.decode.v1";
    bool metadata_only = false;
    bool all_moe_stages = false;
    bool all_attention_stages = false;
    bool all_attention_o_stages = false;
};

static uint64_t qwen3moe_stages_per_rank(const qwen3moe_context_state & state, uint32_t n_layer, bool output) {
    // Output selection follows attention, so only the final MoE region is pruned.
    return uint64_t(n_layer) *
        (1 + unsigned(state.all_attention_stages) + unsigned(state.all_attention_o_stages)) - (output ? 0 : 1);
}

bool qwen3moe_model_init(llama_model * model) {
    return model->arch == LLM_ARCH_QWEN3MOE;
}

bool qwen3moe_context_init(llama_context * ctx) {
    auto state = std::make_unique<qwen3moe_context_state>();
    const auto & model = ctx->get_model();
    const auto & hparams = model.hparams;

    const char * variant = std::getenv("LLAMA_EXECUTION_EXTENSION_VARIANT");
    if (variant != nullptr && variant[0] != '\0' && std::strcmp(variant, "optimized") != 0) {
        if (std::strcmp(variant, "gate-up-reference") == 0) {
            state->plan_name = "qwen3moe.decode.gate-up-reference.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "metadata-only") == 0) {
            state->plan_name = "qwen3moe.decode.metadata-only.v1";
            state->metadata_only = true;
        } else if (std::strcmp(variant, "combined-stages") == 0) {
            state->plan_name = "qwen3moe.decode.combined-stages.v1";
        } else if (std::strcmp(variant, "full-moe") == 0) {
            state->plan_name = "qwen3moe.decode.full-moe.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "full-moe-q8-256") == 0) {
            state->plan_name = "qwen3moe.decode.full-moe-q8-256.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "full-moe-q8-128") == 0) {
            state->plan_name = "qwen3moe.decode.full-moe-q8-128.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "full-ffn-q8-128") == 0) {
            state->plan_name = "qwen3moe.decode.full-ffn-q8-128.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "split-moe-q8-128") == 0) {
            state->plan_name = "qwen3moe.decode.split-moe-q8-128.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "full-ffn-q8-128-direct") == 0) {
            state->plan_name = "qwen3moe.decode.full-ffn-q8-128-direct.v1";
            state->all_moe_stages = true;
        } else if (std::strcmp(variant, "split-moe-qkv-prefix-q8-128") == 0) {
            state->plan_name = "qwen3moe.decode.split-moe-qkv-prefix-q8-128.v1";
            state->all_moe_stages = true;
            state->all_attention_stages = true;
        } else if (std::strcmp(variant, "split-moe-qkv-prefix-o-q8-128") == 0) {
            state->plan_name = "qwen3moe.decode.split-moe-qkv-prefix-o-q8-128.v1";
            state->all_moe_stages = true;
            state->all_attention_stages = true;
            state->all_attention_o_stages = true;
        } else {
            LLAMA_LOG_ERROR("%s: unknown execution extension variant: %s\n", __func__, variant);
            return false;
        }
    }

    state->supported_model =
        hparams.n_layer() == 94 && hparams.n_embd == 4096 &&
        hparams.n_expert == 128 && hparams.n_expert_used == 8 &&
        model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
        model.n_gpu_layers() > hparams.n_layer_all &&
        !model.has_tensor_overrides();

    if (state->supported_model) {
        auto sched = ctx->get_sched();
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto backend = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_meta_execution_extension_begin(backend, state->plan_name)) {
                if (!ggml_backend_meta_execution_extension_end(backend, nullptr)) {
                    return false;
                }
                state->backend = backend;
                break;
            }
        }
    }

    const char * stats_env = std::getenv("LLAMA_EXECUTION_EXTENSION_STATS");
    if (stats_env != nullptr && std::strcmp(stats_env, "1") == 0) {
        std::fprintf(stderr, "EXECUTION_EXTENSION_PLAN=%s\n", state->plan_name);
        std::fflush(stderr);
    } else {
        LLAMA_LOG_INFO("%s: decode plan %s; backend adapter %s; prefill uses stock execution\n", __func__,
                state->plan_name, state->backend != nullptr ? "available" : "unavailable (stock fallback)");
    }

    ctx->execution_extension_state = state.release();
    return true;
}

void qwen3moe_context_free(llama_context * ctx) {
    delete static_cast<qwen3moe_context_state *>(ctx->execution_extension_state);
    ctx->execution_extension_state = nullptr;
}

bool qwen3moe_supports(const llama_context * ctx, const llama_ubatch & ubatch, const llama_memory_context_i * mctx) {
    const auto * state = static_cast<const qwen3moe_context_state *>(ctx->execution_extension_state);
    const auto & cparams = ctx->get_cparams();

    return state != nullptr && state->supported_model && mctx != nullptr &&
           ubatch.n_tokens == 1 && ubatch.n_seqs_unq == 1 && ubatch.n_pos == 1 &&
           ubatch.token != nullptr && ubatch.embd == nullptr && ubatch.output != nullptr &&
           ubatch.n_seq_id != nullptr && ubatch.n_seq_id[0] == 1 &&
           cparams.causal_attn && cparams.flash_attn && cparams.offload_kqv &&
           !cparams.pipeline_parallel;
}

llama_execution_extension_result qwen3moe_execute(llama_context * ctx, const llama_ubatch & ubatch, llama_memory_context_i * mctx) {
    const auto * state = static_cast<const qwen3moe_context_state *>(ctx->execution_extension_state);
    if (state->backend == nullptr || !ggml_backend_meta_execution_extension_begin(state->backend, state->plan_name)) {
        return {};
    }

    llama_execution_extension_result result;
    result.handled = true;
    try {
        result.graph = ctx->process_ubatch(ubatch, LLM_GRAPH_TYPE_DEFAULT, mctx, result.status);
    } catch (...) {
        ggml_backend_meta_execution_extension_end(state->backend, nullptr);
        throw;
    }

    if (!ggml_backend_meta_execution_extension_end(state->backend, &result.custom_stages) ||
        (state->metadata_only && result.custom_stages != 0)) {
        result.graph = nullptr;
        result.status = GGML_STATUS_FAILED;
    } else if (state->metadata_only && result.graph != nullptr && result.status == GGML_STATUS_SUCCESS) {
        result.metadata_only = true;
    } else if (state->all_moe_stages && result.graph != nullptr && result.status == GGML_STATUS_SUCCESS) {
        const uint64_t stages_per_rank = qwen3moe_stages_per_rank(
            *state, ctx->get_model().hparams.n_layer(), ubatch.output[0] != 0);
        const uint64_t expected = stages_per_rank * ggml_backend_meta_n_backends(state->backend);
        if (result.custom_stages != expected) {
            LLAMA_LOG_ERROR("%s: incomplete MoE coverage: custom_stages=%" PRIu64 " expected=%" PRIu64 "\n",
                    __func__, result.custom_stages, expected);
            result.graph = nullptr;
            result.status = GGML_STATUS_FAILED;
        }
    }
    return result;
}

} // namespace

const llama_execution_extension qwen3moe_execution_extension = {
    "qwen3moe.decode",
    qwen3moe_model_init,
    qwen3moe_context_init,
    qwen3moe_context_free,
    qwen3moe_supports,
    qwen3moe_execute,
};
