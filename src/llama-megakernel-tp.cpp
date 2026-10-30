#include "llama-megakernel-tp.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "llama-batch.h"
#include "llama-kv-cache.h"

#include <cstdlib>
#include <cstring>
#include <vector>

// the kernel's build: Qwen3-235B-A22B, 3 devices
static constexpr int32_t MK_TP_DEVICES   = 3;
static constexpr int32_t MK_TP_LAYERS    = 94;
static constexpr int32_t MK_TP_PER_LAYER = 14; // the order of RPC_MK_PER_LAYER in ggml-rpc.cpp

typedef bool (*mk_init_fn_t)(ggml_backend_t *, size_t, struct ggml_tensor * const *, size_t, const int32_t *, char *, size_t);
typedef bool (*mk_decode_fn_t)(ggml_backend_t *, size_t, int32_t, int32_t, const void *, size_t, float * const *, const int32_t *);

struct llama_megakernel_tp {
    std::vector<ggml_backend_t> backends;      // the RPC devices, in llama.cpp's order
    std::vector<int32_t>        rows;          // LM head rows per device
    std::vector<int32_t>        vocab_off;     // and the first one's vocabulary index
    std::vector<ggml_tensor *>  k, v;          // KV view scratch
    const ggml_tensor *         tok_embd = nullptr; // on the host: the token's row goes with each step
    std::vector<uint8_t>        emb_row;
    mk_decode_fn_t              decode = nullptr;
    uint64_t                    n_done = 0, n_graph = 0;
};

void llama_megakernel_tp_deleter::operator()(llama_megakernel_tp * mk) const {
    if (mk) {
        LLAMA_LOG_INFO("llama_megakernel_tp: %llu single-token ubatches decoded by the kernel, %llu by the graph\n",
                       (unsigned long long) mk->n_done, (unsigned long long) mk->n_graph);
    }
    delete mk;
}

static const char * mk_tp_unsupported(const llama_model & model, const llama_cparams & cparams, ggml_backend_t meta,
                                      const llama_kv_cache * kv) {
    const auto & hp = model.hparams;
    // fewer layers: a truncated model (tools/truncate_gguf.py) for tests
    if (model.arch != LLM_ARCH_QWEN3MOE || hp.n_layer() > MK_TP_LAYERS || hp.n_embd != 4096 || hp.n_head() != 64 ||
        hp.n_head_kv() != 4 || hp.n_embd_head_k() != 128 || hp.n_expert != 128 || hp.n_expert_used != 8 ||
        hp.n_ff_exp != 1536) {
        return "the kernel is built for Qwen3-235B-A22B";
    }
    if (model.split_mode() != LLAMA_SPLIT_MODE_TENSOR || meta == nullptr ||
        ggml_backend_meta_n_backends(meta) != (size_t) MK_TP_DEVICES) {
        return "needs -sm tensor over 3 devices";
    }
    if (kv == nullptr) {
        return "needs the unified KV cache";
    }
    if (!cparams.flash_attn) {
        return "needs flash attention (V not transposed)";
    }
    if (cparams.rope_freq_base != 1000000.0f || cparams.rope_freq_scale != 1.0f || model.output == nullptr) {
        return "rope / output settings differ from the kernel's";
    }
    return nullptr;
}

llama_megakernel_tp_ptr llama_megakernel_tp_init(const llama_model & model, const llama_cparams & cparams,
                                                 ggml_backend_t meta_backend, llama_memory_i * memory) {
    const char * env = std::getenv("LLAMA_MEGAKERNEL");
    if (env == nullptr || std::atoi(env) == 0) {
        return nullptr;
    }
    const auto * kv = dynamic_cast<const llama_kv_cache *>(memory);
    if (const char * why = mk_tp_unsupported(model, cparams, meta_backend, kv)) {
        LLAMA_LOG_WARN("%s: not used: %s\n", __func__, why);
        return nullptr;
    }
    ggml_backend_reg_t rpc = ggml_backend_reg_by_name("RPC");
    auto init_fn = rpc ? (mk_init_fn_t) ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_mk_init") : nullptr;
    auto dec_fn  = rpc ? (mk_decode_fn_t) ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_mk_decode") : nullptr;
    if (!init_fn || !dec_fn) {
        LLAMA_LOG_WARN("%s: not used: the RPC backend has no megakernel commands\n", __func__);
        return nullptr;
    }

    llama_megakernel_tp_ptr mk(new llama_megakernel_tp());
    const int32_t n_layer = (int32_t) model.hparams.n_layer();
    for (int32_t j = 0; j < MK_TP_DEVICES; j++) {
        mk->backends.push_back(ggml_backend_meta_simple_backend(meta_backend, j));
    }

    // every device's slice of every tensor, in the order of RPC_MK_PER_LAYER
    std::vector<ggml_tensor *> full;
    for (int32_t il = 0; il < n_layer; il++) {
        const auto & L = model.layers[il];
        for (ggml_tensor * t : {L.wq, L.wk, L.wv, L.wo, L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps, L.ffn_gate_inp,
                                L.attn_norm, L.ffn_norm, L.attn_q_norm, L.attn_k_norm, kv->get_layer_k(il), kv->get_layer_v(il)}) {
            full.push_back(t);
        }
    }
    full.push_back(model.output);
    full.push_back(model.output_norm);
    const ggml_tensor * emb = model.tok_embd;
    if (emb == nullptr || emb->buffer == nullptr || ggml_backend_buffer_is_meta(emb->buffer) || emb->ne[2] != 1) {
        LLAMA_LOG_WARN("%s: not used: token_embd is expected in host memory\n", __func__);
        return nullptr;
    }
    mk->tok_embd = emb;
    mk->emb_row.resize(emb->nb[1]);
    std::vector<ggml_tensor *> slices;
    for (int32_t j = 0; j < MK_TP_DEVICES; j++) {
        for (ggml_tensor * t : full) {
            ggml_tensor * s = t && t->buffer && ggml_backend_buffer_is_meta(t->buffer) ? ggml_backend_meta_tensor_simple(t, j) : nullptr;
            if (s == nullptr) {
                LLAMA_LOG_WARN("%s: not used: %s has no slice on device %d\n", __func__, t ? t->name : "a tensor", j);
                return nullptr;
            }
            if (!ggml_is_contiguous(s)) {
                LLAMA_LOG_WARN("%s: not used: the slice of %s on device %d is not contiguous\n", __func__, t->name, j);
                return nullptr;
            }
            slices.push_back(s);
        }
    }

    // the LM head's vocabulary rows: the split rotates which device gets which size, but the slices follow the
    // device order along the tensor
    int32_t off = 0;
    for (int32_t j = 0; j < MK_TP_DEVICES; j++) {
        mk->rows.push_back((int32_t) slices[(size_t) j * full.size() + full.size() - 2]->ne[1]);
        mk->vocab_off.push_back(off);
        off += mk->rows.back();
    }
    std::vector<int32_t> hdr;
    for (int32_t j = 0; j < MK_TP_DEVICES; j++) {
        hdr.insert(hdr.end(), {j, MK_TP_DEVICES, n_layer, (int32_t) kv->get_size(), mk->vocab_off[j], (int32_t) emb->type,
                               (int32_t) emb->ne[0], (int32_t) emb->ne[1]});
    }
    char err[300] = "";
    const int64_t t0 = ggml_time_us();
    if (!init_fn(mk->backends.data(), mk->backends.size(), slices.data(), full.size(), hdr.data(), err, sizeof(err))) {
        LLAMA_LOG_WARN("%s: not used: %s\n", __func__, err);
        return nullptr;
    }
    mk->decode = dec_fn;
    mk->k.resize(n_layer);
    mk->v.resize(n_layer);
    LLAMA_LOG_INFO("%s: Qwen3-235B-A22B tensor-split decode kernel on the 3 RPC devices (set up in %.1f s)\n",
                   __func__, (ggml_time_us() - t0) / 1e6);
    return mk;
}

llama_megakernel_tp_result llama_megakernel_tp_decode(llama_megakernel_tp * mk, const llama_ubatch & ubatch,
                                                      llama_memory_context_i * mctx, ggml_backend_sched_t sched,
                                                      float * logits_out) {
    auto * kvc = dynamic_cast<llama_kv_cache_context *>(mctx);
    if (mk->decode == nullptr || kvc == nullptr || !kvc->get_megakernel_tp_view(mk->k, mk->v)) {
        mk->n_graph++;
        return LLAMA_MEGAKERNEL_TP_SKIPPED;
    }
    ggml_backend_sched_synchronize(sched);
    if (!mctx->apply()) {
        mk->n_graph++;
        return LLAMA_MEGAKERNEL_TP_SKIPPED;
    }
    std::vector<float *> out(MK_TP_DEVICES, nullptr);
    for (int32_t j = 0; j < MK_TP_DEVICES && logits_out; j++) {
        out[j] = logits_out + mk->vocab_off[j];
    }
    const llama_token tok = ubatch.token[0];
    if (tok < 0 || tok >= mk->tok_embd->ne[1]) {
        return LLAMA_MEGAKERNEL_TP_APPLIED;
    }
    ggml_backend_tensor_get(mk->tok_embd, mk->emb_row.data(), (size_t) tok * mk->tok_embd->nb[1], mk->emb_row.size());
    if (!mk->decode(mk->backends.data(), mk->backends.size(), tok, ubatch.pos[0], mk->emb_row.data(), mk->emb_row.size(),
                    out.data(), mk->rows.data())) {
        LLAMA_LOG_ERROR("%s: the decode kernel failed; continuing with the graph from here on\n", __func__);
        mk->decode = nullptr;
        mk->n_graph++;
        return LLAMA_MEGAKERNEL_TP_APPLIED;
    }
    mk->n_done++;
    return LLAMA_MEGAKERNEL_TP_DONE;
}
