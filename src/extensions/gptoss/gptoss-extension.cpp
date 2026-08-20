#include "gptoss-extension.h"

#include "extensions/hip-workspace.h"
#include "extensions/llama-execution-extension.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gptoss-aot.h"
#include "gptoss-config.h"
#include "gptoss-kernel.h"
#include "gptoss-repack-hip.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model.h"

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#define GPTOSS_AOT_DIR "/app/llama.cpp/src/extensions/gptoss/build/gfx1201"

namespace {

constexpr uint32_t gptoss_max_attention_parts = 12;

constexpr const char * gptoss_fa_name = "kernel_unified_attention_2d";

struct gptoss_context_state {
    hipStream_t stream = nullptr;

    hipModule_t q8_qkv_module      = nullptr;
    hipModule_t q8_attn_out_module = nullptr;
    hipModule_t fa_full_module     = nullptr;
    hipModule_t fa_swa_module      = nullptr;
    hipModule_t ogs_w13_module     = nullptr;
    hipModule_t ogs_w2_module      = nullptr;

    hipFunction_t q8_qkv      = nullptr;
    hipFunction_t q8_attn_out = nullptr;
    hipFunction_t fa_full     = nullptr;
    hipFunction_t fa_swa      = nullptr;
    hipFunction_t ogs_w13     = nullptr;
    hipFunction_t ogs_w2      = nullptr;

    hipblasHandle_t hipblas = nullptr;

    llama_hip_workspace workspace;
};

bool gptoss_hip_ok(hipError_t error, const char * operation) {
    if (error == hipSuccess) {
        return true;
    }

    LLAMA_LOG_ERROR("gptoss: %s failed: %s\n", operation, hipGetErrorString(error));
    return false;
}

struct gptoss_stream_guard {
    explicit gptoss_stream_guard(hipStream_t stream) : stream(stream) {}

    ~gptoss_stream_guard() {
        if (!synchronized) {
            (void) hipStreamSynchronize(stream);
        }
    }

    hipError_t synchronize() {
        synchronized = true;
        return hipStreamSynchronize(stream);
    }

    hipStream_t stream;
    bool synchronized = false;
};

size_t gptoss_tensor_alloc_size(const ggml_tensor * tensor);

struct gptoss_sequence_span {
    llama_seq_id sequence;
    uint32_t     begin;
    uint32_t     size;
};

bool gptoss_sequence_spans(const llama_ubatch & ubatch, std::vector<gptoss_sequence_span> & spans) {
    if (ubatch.n_tokens == 0 || ubatch.n_pos != 1) {
        return false;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] <= 0 || ubatch.pos[i] < 0) {
            return false;
        }

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            if (ubatch.seq_id[i][s] < 0) {
                return false;
            }
        }

        const llama_seq_id sequence = ubatch.seq_id[i][0];
        if (spans.empty() || spans.back().sequence != sequence) {
            for (const auto & span : spans) {
                if (span.sequence == sequence) {
                    return false;
                }
            }
            spans.push_back({ sequence, i, 1 });
        } else {
            if (ubatch.pos[i] != ubatch.pos[i - 1] + 1 || ubatch.n_seq_id[i] != ubatch.n_seq_id[i - 1]) {
                return false;
            }
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                if (ubatch.seq_id[i][s] != ubatch.seq_id[i - 1][s]) {
                    return false;
                }
            }
            spans.back().size++;
        }
    }

    return true;
}

struct gptoss_kv_layout {
    const llama_kv_cache * cache = nullptr;

    std::vector<int64_t> write_rows;
    std::vector<int32_t> read_rows;
    std::vector<int32_t> seq_lens;
};

bool gptoss_build_kv_layout(const llama_ubatch &                      ubatch,
                            const std::vector<gptoss_sequence_span> & spans,
                            const llama_kv_cache_context *            context,
                            bool                                      sliding_window,
                            gptoss_kv_layout &                        layout) {
    const auto * cache = context->get_cache();
    const auto & slot  = context->get_slot_info();

    if (cache == nullptr || cache->get_has_shift() || slot.empty() || slot.n_stream() * slot.size() != ubatch.n_tokens) {
        return false;
    }

    const uint32_t cache_size = cache->get_size();
    const uint32_t n_streams  = cache->get_n_stream();
    layout.cache              = cache;
    layout.write_rows.resize(ubatch.n_tokens);

    for (uint32_t stream = 0; stream < slot.n_stream(); ++stream) {
        for (uint32_t token = 0; token < slot.size(); ++token) {
            const uint32_t i = stream * slot.size() + token;
            if (static_cast<uint32_t>(slot.strm[stream]) >= n_streams || slot.idxs[stream][token] >= cache_size) {
                return false;
            }
            layout.write_rows[i] = static_cast<int64_t>(slot.strm[stream]) * cache_size + slot.idxs[stream][token];
        }
    }

    struct cell_row {
        llama_pos position;
        int32_t   row;
    };

    for (const auto & span : spans) {
        const uint32_t stream = static_cast<uint32_t>(layout.write_rows[span.begin] / cache_size);
        for (uint32_t i = 1; i < span.size; ++i) {
            if (static_cast<uint32_t>(layout.write_rows[span.begin + i] / cache_size) != stream) {
                return false;
            }
        }

        const llama_pos first_position = ubatch.pos[span.begin];
        const llama_pos last_position  = ubatch.pos[span.begin + span.size - 1];
        const llama_pos first_visible =
            sliding_window ? first_position - static_cast<llama_pos>(gptoss_swa_size - 1) : 0;
        const auto &    cells          = cache->get_cells(span.sequence);

        std::vector<cell_row> rows;
        rows.reserve(cells.get_used());
        for (uint32_t row = 0; row < cells.size(); ++row) {
            if (cells.is_empty(row) || !cells.seq_has(row, span.sequence)) {
                continue;
            }

            const llama_pos position = cells.pos_get(row);
            if (position <= last_position && position >= first_visible) {
                const uint64_t global_row = static_cast<uint64_t>(stream) * cache_size + row;
                if (global_row > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                    return false;
                }
                rows.push_back({ position, static_cast<int32_t>(global_row) });
            }
        }

        std::sort(rows.begin(), rows.end(),
                  [](const cell_row & a, const cell_row & b) { return a.position < b.position; });

        if (rows.size() < span.size || rows.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        for (size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].position != rows[i - 1].position + 1) {
                return false;
            }
        }
        for (uint32_t i = 0; i < span.size; ++i) {
            const auto & row = rows[rows.size() - span.size + i];
            if (row.position != ubatch.pos[span.begin + i] || row.row != layout.write_rows[span.begin + i]) {
                return false;
            }
        }

        for (const auto & row : rows) {
            layout.read_rows.push_back(row.row);
        }
        layout.seq_lens.push_back(static_cast<int32_t>(rows.size()));
    }

    return !layout.read_rows.empty();
}

struct gptoss_fa_layout {
    std::vector<int32_t> block_table;
    int64_t              block_table_stride = 0;
};

bool gptoss_build_fa_layout(const gptoss_kv_layout & kv, gptoss_fa_layout & layout) {
    if (kv.seq_lens.empty()) {
        return false;
    }

    const size_t n_sequences = kv.seq_lens.size();
    size_t       n_rows      = 0;
    int32_t      max_seq_len = 0;

    for (const int32_t seq_len : kv.seq_lens) {
        if (seq_len <= 0 || static_cast<size_t>(seq_len) > kv.read_rows.size() - n_rows) {
            return false;
        }
        n_rows += static_cast<size_t>(seq_len);
        max_seq_len = std::max(max_seq_len, seq_len);
    }
    if (n_rows != kv.read_rows.size()) {
        return false;
    }

    if (max_seq_len > std::numeric_limits<int32_t>::max() - static_cast<int32_t>(gptoss_fa_tile_size - 1)) {
        return false;
    }

    layout.block_table_stride = (max_seq_len + gptoss_fa_tile_size - 1) / gptoss_fa_tile_size * gptoss_fa_tile_size;
    const size_t block_table_stride = static_cast<size_t>(layout.block_table_stride);
    if (block_table_stride > std::numeric_limits<uint32_t>::max() / n_sequences) {
        return false;
    }
    layout.block_table.resize(block_table_stride * n_sequences);

    size_t begin = 0;
    for (size_t i = 0; i < n_sequences; ++i) {
        const size_t end = begin + static_cast<size_t>(kv.seq_lens[i]);
        int32_t *    dst   = layout.block_table.data() + i * block_table_stride;
        std::copy(kv.read_rows.begin() + begin, kv.read_rows.begin() + end, dst);
        std::fill(dst + end - begin, dst + block_table_stride, kv.read_rows[end - 1]);
        begin = end;
    }

    return true;
}

bool gptoss_get_kv(const llama_kv_cache * cache, uint32_t layer, __half *& k, __half *& v) {
    const ggml_tensor * tensor_k = cache->get_k_storage(layer);
    const ggml_tensor * tensor_v = cache->get_v_storage(layer);

    if (tensor_k->type != GGML_TYPE_F16 || tensor_v->type != GGML_TYPE_F16 ||
        tensor_k->ne[0] != gptoss_key_value_size || tensor_v->ne[0] != gptoss_key_value_size ||
        tensor_k->ne[1] != cache->get_size() || tensor_v->ne[1] != cache->get_size() ||
        tensor_k->ne[2] != cache->get_n_stream() || tensor_v->ne[2] != cache->get_n_stream() ||
        tensor_k->nb[1] != gptoss_key_value_size * sizeof(uint16_t) ||
        tensor_v->nb[1] != gptoss_key_value_size * sizeof(uint16_t) ||
        tensor_k->nb[2] != cache->get_size() * tensor_k->nb[1] ||
        tensor_v->nb[2] != cache->get_size() * tensor_v->nb[1]) {
        return false;
    }

    k = static_cast<__half *>(tensor_k->data);
    v = static_cast<__half *>(tensor_v->data);
    return k != nullptr && v != nullptr;
}

struct gptoss_prefill_buffers {
    int32_t * tokens;
    int32_t * positions;

    int64_t * base_write_rows;
    int64_t * swa_write_rows;
    int32_t * base_block_table;
    int32_t * swa_block_table;
    int32_t * cu_seqlens_q;
    int32_t * base_seq_lens;
    int32_t * swa_seq_lens;

    float *  cur;
    float *  next;
    __half * norm;
    float *  rope_base;
    float *  rope_swa;
    __half * qkv;
    __half * q;

    float *   router_logits;
    int32_t * selected_ids;
    float *   selected_weights;
    int32_t * gather_indices;
    int32_t * scatter_indices;
    int32_t * expert_counts;
    int32_t * route_offsets;
    int32_t * block_offsets;
    int32_t * block_schedule;
    __half *  expert_activations;
    float *   expert_outputs;

    float * logits;
};

gptoss_prefill_buffers gptoss_make_prefill_buffers(llama_hip_workspace_cursor & cursor,
                                                   uint32_t                     n_tokens,
                                                   bool                         output,
                                                   uint32_t                     n_sequences,
                                                   uint32_t                     n_base_table_elements,
                                                   uint32_t                     n_swa_table_elements) {
    const uint32_t route_count = n_tokens * gptoss_expert_used_count;
    const uint32_t schedule_capacity =
        (route_count + gptoss_ogs_block_m - 1) / gptoss_ogs_block_m + gptoss_expert_count - 1;
    gptoss_prefill_buffers result = {};

    result.tokens             = cursor.take<int32_t>(n_tokens);
    result.positions          = cursor.take<int32_t>(n_tokens);
    result.base_write_rows    = cursor.take<int64_t>(n_tokens);
    result.swa_write_rows     = cursor.take<int64_t>(n_tokens);
    result.base_block_table   = cursor.take<int32_t>(n_base_table_elements);
    result.swa_block_table    = cursor.take<int32_t>(n_swa_table_elements);
    result.cu_seqlens_q       = cursor.take<int32_t>(n_sequences + 1);
    result.base_seq_lens      = cursor.take<int32_t>(n_sequences);
    result.swa_seq_lens       = cursor.take<int32_t>(n_sequences);
    result.cur                = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.next               = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.norm               = cursor.take<__half>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.rope_base          = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_head_size);
    result.rope_swa           = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_head_size);
    result.qkv                = cursor.take<__half>(static_cast<size_t>(n_tokens) * gptoss_qkv_size);
    result.q                  = cursor.take<__half>(static_cast<size_t>(n_tokens) * gptoss_query_size);
    result.router_logits      = cursor.take<float>(static_cast<size_t>(n_tokens) * gptoss_expert_count);
    result.selected_ids       = cursor.take<int32_t>(route_count);
    result.selected_weights   = cursor.take<float>(route_count);
    result.gather_indices     = cursor.take<int32_t>(route_count);
    result.scatter_indices    = cursor.take<int32_t>(route_count);
    result.expert_counts      = cursor.take<int32_t>(gptoss_expert_count);
    result.route_offsets      = cursor.take<int32_t>(gptoss_expert_count + 1);
    result.block_offsets      = cursor.take<int32_t>(gptoss_expert_count + 1);
    result.block_schedule     = cursor.take<int32_t>(schedule_capacity);
    result.expert_activations = cursor.take<__half>(static_cast<size_t>(route_count) * gptoss_intermediate_size);
    result.expert_outputs     = cursor.take<float>(static_cast<size_t>(route_count) * gptoss_hidden_size);
    result.logits             = cursor.take<float>(output ? gptoss_vocabulary_size : 0);

    return result;
}

struct gptoss_decode_buffers {
    int32_t * token;
    int32_t * base_rows;
    int32_t * swa_rows;

    float *   cur;
    float *   next;
    float *   rms_partials;
    __half *  activation_scratch;
    __half *  query;
    float *   router_scores;
    int32_t * selected_experts;
    float *   selected_weights;
    float *   attention_parts;
    float2 *  attention_meta;
    uint32_t  attention_partitions;

    float * logits;
};

gptoss_decode_buffers gptoss_make_decode_buffers(llama_hip_workspace_cursor & cursor,
                                                 size_t                       n_base_rows,
                                                 size_t                       n_swa_rows,
                                                 bool                         output) {
    const uint32_t partitions = static_cast<uint32_t>(std::min<size_t>(
        gptoss_max_attention_parts,
        std::max<size_t>(2, (n_base_rows + gptoss_swa_size - 1) / gptoss_swa_size)));

    gptoss_decode_buffers result = {};

    result.token            = cursor.take<int32_t>(1);
    result.base_rows        = cursor.take<int32_t>(n_base_rows + n_swa_rows);
    result.swa_rows         = result.base_rows == nullptr ? nullptr : result.base_rows + n_base_rows;
    result.cur              = cursor.take<float>(gptoss_hidden_size);
    result.next             = cursor.take<float>(gptoss_hidden_size);
    result.rms_partials     = cursor.take<float>(gptoss_decode_grid_blocks);
    result.activation_scratch =
        cursor.take<__half>(gptoss_hidden_size * (1 + gptoss_expert_used_count));
    result.query            = cursor.take<__half>(gptoss_query_size);
    result.router_scores    = cursor.take<float>(gptoss_expert_count);
    result.selected_experts = cursor.take<int32_t>(gptoss_expert_used_count);
    result.selected_weights = cursor.take<float>(gptoss_expert_used_count);
    result.attention_parts =
        cursor.take<float>(static_cast<size_t>(gptoss_query_head_count) * partitions * gptoss_head_size);
    result.attention_meta       = cursor.take<float2>(static_cast<size_t>(gptoss_query_head_count) * partitions);
    result.attention_partitions = partitions;
    result.logits               = cursor.take<float>(output ? gptoss_vocabulary_size : 0);

    return result;
}

void gptoss_context_free(llama_context * ctx) {
    auto * state = static_cast<gptoss_context_state *>(ctx->execution_extension_state);

    if (state == nullptr) {
        return;
    }

    if (state->stream != nullptr) {
        (void) hipStreamSynchronize(state->stream);
    }

    if (state->hipblas != nullptr) {
        (void) hipblasDestroy(state->hipblas);
    }
    (void) gptoss_hip_ok(state->workspace.free(), "workspace free");

    if (state->q8_qkv_module != nullptr) {
        (void) hipModuleUnload(state->q8_qkv_module);
    }
    if (state->q8_attn_out_module != nullptr) {
        (void) hipModuleUnload(state->q8_attn_out_module);
    }
    if (state->fa_full_module != nullptr) {
        (void) hipModuleUnload(state->fa_full_module);
    }
    if (state->fa_swa_module != nullptr) {
        (void) hipModuleUnload(state->fa_swa_module);
    }
    if (state->ogs_w13_module != nullptr) {
        (void) hipModuleUnload(state->ogs_w13_module);
    }
    if (state->ogs_w2_module != nullptr) {
        (void) hipModuleUnload(state->ogs_w2_module);
    }
    if (state->stream != nullptr) {
        (void) hipStreamDestroy(state->stream);
    }
    delete state;
    ctx->execution_extension_state = nullptr;
}

bool gptoss_model_init(llama_model * model) {
    const llama_hparams & hparams = model->hparams;

    if (model->n_devices() != 1 || model->n_gpu_layers() <= hparams.n_layer_all || model->has_tensor_overrides() ||
        hparams.n_layer_all != gptoss_layer_count || hparams.n_embd != gptoss_hidden_size ||
        hparams.n_ff_exp != gptoss_intermediate_size || hparams.n_expert != gptoss_expert_count ||
        hparams.n_expert_used != gptoss_expert_used_count || hparams.n_expert_groups != 0 ||
        hparams.n_group_used != 0 || hparams.n_swa != gptoss_swa_size ||
        model->vocab.n_tokens() != gptoss_vocabulary_size) {
        LLAMA_LOG_ERROR("%s: unsupported model hyperparameters\n", __func__);
        return false;
    }

    int             hip_device_count = 0;
    hipDeviceProp_t properties{};
    if (hipGetDeviceCount(&hip_device_count) != hipSuccess || hip_device_count != 1 ||
        ggml_backend_cuda_get_device_count() != 1 || hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        properties.warpSize != 32 || !properties.cooperativeLaunch) {
        LLAMA_LOG_ERROR("%s: unsupported device\n", __func__);
        return false;
    }

    const auto hip_buft = ggml_backend_cuda_buffer_type(0);
    const auto device_tensor_is = [hip_buft](const ggml_tensor * tensor, ggml_type type) {
        return tensor != nullptr && tensor->type == type && tensor->data != nullptr && tensor->buffer != nullptr &&
               ggml_backend_buffer_get_type(tensor->buffer) == hip_buft;
    };
    const auto follows = [](const ggml_tensor * first, const ggml_tensor * second) {
        if (first->buffer != second->buffer) {
            return false;
        }

        const auto buft      = ggml_backend_buffer_get_type(first->buffer);
        const auto alignment = ggml_backend_buft_get_alignment(buft);
        const auto size      = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, first), alignment);
        return reinterpret_cast<uintptr_t>(second->data) == reinterpret_cast<uintptr_t>(first->data) + size;
    };

    if (!device_tensor_is(model->tok_embd, GGML_TYPE_Q8_0) || !device_tensor_is(model->output, GGML_TYPE_Q8_0) ||
        !device_tensor_is(model->output_norm, GGML_TYPE_F32)) {
        LLAMA_LOG_ERROR("%s: unsupported model tensors\n", __func__);
        return false;
    }

    for (uint32_t il = 0; il < gptoss_layer_count; ++il) {
        const llama_layer & layer = model->layers[il];

        if (hparams.is_swa(il) != (il % 2 == 0) || hparams.n_head(il) != gptoss_query_head_count ||
            hparams.n_head_kv(il) != gptoss_kv_head_count || hparams.n_rot(il) != gptoss_head_size ||
            hparams.n_embd_head_k(il) != gptoss_head_size || hparams.n_embd_head_v(il) != gptoss_head_size) {
            LLAMA_LOG_ERROR("%s: unsupported layer %u hyperparameters\n", __func__, il);
            return false;
        }

        if (!device_tensor_is(layer.wq, GGML_TYPE_Q8_0) || !device_tensor_is(layer.wk, GGML_TYPE_Q8_0) ||
            !device_tensor_is(layer.wv, GGML_TYPE_Q8_0) || !device_tensor_is(layer.wo, GGML_TYPE_Q8_0) ||
            !device_tensor_is(layer.ffn_gate_exps, GGML_TYPE_MXFP4) ||
            !device_tensor_is(layer.ffn_down_exps, GGML_TYPE_MXFP4) ||
            !device_tensor_is(layer.ffn_up_exps, GGML_TYPE_MXFP4) ||
            !device_tensor_is(layer.attn_norm, GGML_TYPE_F32) ||
            !device_tensor_is(layer.attn_post_norm, GGML_TYPE_F32) || !device_tensor_is(layer.wq_b, GGML_TYPE_F32) ||
            !device_tensor_is(layer.wk_b, GGML_TYPE_F32) || !device_tensor_is(layer.wv_b, GGML_TYPE_F32) ||
            !device_tensor_is(layer.wo_b, GGML_TYPE_F32) || !device_tensor_is(layer.attn_sinks, GGML_TYPE_F32) ||
            !device_tensor_is(layer.ffn_gate_inp, GGML_TYPE_F32) ||
            !device_tensor_is(layer.ffn_gate_inp_b, GGML_TYPE_F32) ||
            !device_tensor_is(layer.ffn_gate_exps_b, GGML_TYPE_F32) ||
            !device_tensor_is(layer.ffn_down_exps_b, GGML_TYPE_F32) ||
            !device_tensor_is(layer.ffn_up_exps_b, GGML_TYPE_F32)) {
            LLAMA_LOG_ERROR("%s: unsupported layer %u tensors\n", __func__, il);
            return false;
        }

        if (!follows(layer.wq, layer.wk) || !follows(layer.wk, layer.wv) || !follows(layer.wq_b, layer.wk_b) ||
            !follows(layer.wk_b, layer.wv_b) || !follows(layer.ffn_gate_exps, layer.ffn_down_exps) ||
            !follows(layer.ffn_down_exps, layer.ffn_up_exps) ||
            !follows(layer.ffn_gate_exps_b, layer.ffn_down_exps_b) ||
            !follows(layer.ffn_down_exps_b, layer.ffn_up_exps_b)) {
            LLAMA_LOG_ERROR("%s: unsupported layer %u tensor layout\n", __func__, il);
            return false;
        }

        const size_t    up_reservation = gptoss_tensor_alloc_size(layer.ffn_up_exps);
        const uintptr_t buffer_base =
            reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(layer.ffn_up_exps->buffer));
        const size_t    buffer_size = ggml_backend_buffer_get_size(layer.ffn_up_exps->buffer);
        const uintptr_t up_data     = reinterpret_cast<uintptr_t>(layer.ffn_up_exps->data);

        if (up_reservation <= ggml_nbytes(layer.ffn_up_exps) || buffer_base == 0 || up_data < buffer_base ||
            up_data - buffer_base > buffer_size || up_reservation > buffer_size - (up_data - buffer_base)) {
            LLAMA_LOG_ERROR("%s: insufficient layer %u expert weight storage\n", __func__, il);
            return false;
        }
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

bool gptoss_context_init(llama_context * ctx) {
    const auto & cparams = ctx->get_cparams();

    if (!cparams.causal_attn || !cparams.flash_attn || !cparams.offload_kqv) {
        return false;
    }

    auto * memory = dynamic_cast<llama_kv_cache_iswa *>(ctx->get_memory());
    if (memory == nullptr || memory->get_base()->type_k() != GGML_TYPE_F16 ||
        memory->get_base()->type_v() != GGML_TYPE_F16 || memory->get_swa()->type_k() != GGML_TYPE_F16 ||
        memory->get_swa()->type_v() != GGML_TYPE_F16) {
        return false;
    }

    auto * state                   = new gptoss_context_state;
    ctx->execution_extension_state = state;

    if (hipStreamCreateWithFlags(&state->stream, hipStreamNonBlocking) != hipSuccess) {
        gptoss_context_free(ctx);
        return false;
    }

    if (hipModuleLoad(&state->q8_qkv_module, GPTOSS_AOT_DIR "/q8_qkv.hsaco") != hipSuccess ||
        hipModuleGetFunction(&state->q8_qkv, state->q8_qkv_module, "gptoss_q8_0_w8a16_qkv_bias") != hipSuccess ||
        hipModuleLoad(&state->q8_attn_out_module, GPTOSS_AOT_DIR "/q8_attn_out.hsaco") != hipSuccess ||
        hipModuleGetFunction(&state->q8_attn_out, state->q8_attn_out_module,
                             "gptoss_q8_0_w8a16_attn_output_bias_residual") != hipSuccess ||
        hipModuleLoad(&state->fa_full_module, GPTOSS_AOT_DIR "/fa_full.hsaco") != hipSuccess ||
        hipModuleGetFunction(&state->fa_full, state->fa_full_module, gptoss_fa_name) != hipSuccess ||
        hipModuleLoad(&state->fa_swa_module, GPTOSS_AOT_DIR "/fa_sw128.hsaco") != hipSuccess ||
        hipModuleGetFunction(&state->fa_swa, state->fa_swa_module, gptoss_fa_name) != hipSuccess ||
        hipModuleLoad(&state->ogs_w13_module, GPTOSS_AOT_DIR "/ogs_w13.hsaco") != hipSuccess ||
        hipModuleGetFunction(&state->ogs_w13, state->ogs_w13_module,
                             "_matmul_NNN_fp16xfp16xmxfp4_64x128x128x1_swiglu") != hipSuccess ||
        hipModuleLoad(&state->ogs_w2_module, GPTOSS_AOT_DIR "/ogs_w2.hsaco") != hipSuccess ||
        hipModuleGetFunction(&state->ogs_w2, state->ogs_w2_module, "_matmul_NNN_fp32xfp16xmxfp4_64x128x128x1") !=
            hipSuccess ||
        hipblasCreate(&state->hipblas) != HIPBLAS_STATUS_SUCCESS ||
        hipblasSetStream(state->hipblas, state->stream) != HIPBLAS_STATUS_SUCCESS) {
        gptoss_context_free(ctx);
        return false;
    }

    return true;
}

int gptoss_prefill(llama_context *                ctx,
                   const llama_ubatch &           ubatch,
                   const llama_memory_context_i * mctx,
                   float *                        logits_out) {
    auto * state = static_cast<gptoss_context_state *>(ctx->execution_extension_state);
    if (state == nullptr || ubatch.token == nullptr || mctx == nullptr ||
        ubatch.n_tokens > std::numeric_limits<int32_t>::max() / gptoss_expert_used_count) {
        return -1;
    }
    std::vector<gptoss_sequence_span> spans;
    if (!gptoss_sequence_spans(ubatch, spans)) {
        LLAMA_LOG_ERROR("%s: unsupported sequence layout\n", __func__);
        return -1;
    }

    const auto *     iswa = static_cast<const llama_kv_cache_iswa_context *>(mctx);
    gptoss_kv_layout base;
    gptoss_kv_layout swa;
    if (!gptoss_build_kv_layout(ubatch, spans, iswa->get_base(), false, base) ||
        !gptoss_build_kv_layout(ubatch, spans, iswa->get_swa(), true, swa)) {
        LLAMA_LOG_ERROR("%s: unsupported KV-cache layout\n", __func__);
        return -1;
    }

    gptoss_fa_layout base_fa;
    gptoss_fa_layout swa_fa;
    if (!gptoss_build_fa_layout(base, base_fa) || !gptoss_build_fa_layout(swa, swa_fa)) {
        LLAMA_LOG_ERROR("%s: unsupported flash-attention layout\n", __func__);
        return -1;
    }

    std::vector<int32_t> cu_seqlens_q(1, 0);
    for (const auto & span : spans) {
        cu_seqlens_q.push_back(cu_seqlens_q.back() + static_cast<int32_t>(span.size));
    }

    bool output = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i] != 0) {
            output = true;
            break;
        }
    }
    if ((logits_out != nullptr) != output) {
        return -1;
    }

    llama_hip_workspace_cursor measure;
    (void) gptoss_make_prefill_buffers(
        measure, ubatch.n_tokens, output, static_cast<uint32_t>(spans.size()),
        static_cast<uint32_t>(base_fa.block_table.size()), static_cast<uint32_t>(swa_fa.block_table.size()));
    if (!measure.valid()) {
        LLAMA_LOG_ERROR("%s: invalid workspace layout\n", __func__);
        return -1;
    }
    if (!gptoss_hip_ok(state->workspace.reserve(measure.size()), "workspace reserve")) {
        return -1;
    }
    llama_hip_workspace_cursor bind(state->workspace.data(), measure.size());
    auto buffers = gptoss_make_prefill_buffers(bind, ubatch.n_tokens, output, static_cast<uint32_t>(spans.size()),
                                               static_cast<uint32_t>(base_fa.block_table.size()),
                                               static_cast<uint32_t>(swa_fa.block_table.size()));
    if (!bind.valid() || bind.size() != measure.size()) {
        LLAMA_LOG_ERROR("%s: invalid workspace layout\n", __func__);
        return -1;
    }

    gptoss_stream_guard stream_guard(state->stream);
    if (!gptoss_hip_ok(hipMemcpyAsync(buffers.tokens, ubatch.token, ubatch.n_tokens * sizeof(int32_t),
                                      hipMemcpyHostToDevice, state->stream),
                       "token upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.positions, ubatch.pos, ubatch.n_tokens * sizeof(int32_t),
                                      hipMemcpyHostToDevice, state->stream),
                       "position upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.base_write_rows, base.write_rows.data(),
                                      base.write_rows.size() * sizeof(int64_t), hipMemcpyHostToDevice, state->stream),
                       "base write-row upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.swa_write_rows, swa.write_rows.data(),
                                      swa.write_rows.size() * sizeof(int64_t), hipMemcpyHostToDevice, state->stream),
                       "SWA write-row upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.base_block_table, base_fa.block_table.data(),
                                      base_fa.block_table.size() * sizeof(int32_t), hipMemcpyHostToDevice,
                                      state->stream),
                       "base block-table upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.swa_block_table, swa_fa.block_table.data(),
                                      swa_fa.block_table.size() * sizeof(int32_t), hipMemcpyHostToDevice,
                                      state->stream),
                       "SWA block-table upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.cu_seqlens_q, cu_seqlens_q.data(), cu_seqlens_q.size() * sizeof(int32_t),
                                      hipMemcpyHostToDevice, state->stream),
                       "query lengths upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.base_seq_lens, base.seq_lens.data(),
                                      base.seq_lens.size() * sizeof(int32_t), hipMemcpyHostToDevice, state->stream),
                       "base lengths upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.swa_seq_lens, swa.seq_lens.data(),
                                      swa.seq_lens.size() * sizeof(int32_t), hipMemcpyHostToDevice, state->stream),
                       "SWA lengths upload")) {
        return -1;
    }

    const auto & model   = ctx->get_model();
    const auto & hparams = model.hparams;
    const auto & cparams = ctx->get_cparams();

    if (!gptoss_hip_ok(
            gptoss_embedding_q8_0_launch(
                buffers.cur, static_cast<const uint8_t *>(model.tok_embd->data), buffers.tokens, ubatch.n_tokens,
                state->stream),
            "token embedding")) {
        return -1;
    }

    auto build_rope = [&](uint32_t layer, float * cache) {
        const float freq_base  = model.get_rope_freq_base(cparams, layer);
        const float freq_scale = model.get_rope_freq_scale(cparams, layer);
        float       corr[2];
        ggml_rope_yarn_corr_dims(gptoss_head_size, cparams.n_ctx_orig_yarn, freq_base, cparams.yarn_beta_fast,
                                 cparams.yarn_beta_slow, corr);
        return gptoss_build_rope_cache_launch(cache, buffers.positions, ubatch.n_tokens, freq_scale,
                                              cparams.yarn_ext_factor, cparams.yarn_attn_factor, corr[0], corr[1],
                                              std::pow(freq_base, -2.0f / gptoss_head_size), state->stream);
    };

    if (!gptoss_hip_ok(build_rope(1, buffers.rope_base), "base RoPE cache") ||
        !gptoss_hip_ok(build_rope(0, buffers.rope_swa), "SWA RoPE cache")) {
        return -1;
    }

    const uint32_t route_count = ubatch.n_tokens * gptoss_expert_used_count;
    const uint32_t schedule_capacity =
        (route_count + gptoss_ogs_block_m - 1) / gptoss_ogs_block_m + gptoss_expert_count - 1;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (uint32_t il = 0; il < gptoss_layer_count; ++il) {
        const llama_layer & layer  = model.layers[il];
        const bool          is_swa = hparams.is_swa(il);
        const auto &        kv     = is_swa ? swa : base;

        __half * cache_k = nullptr;
        __half * cache_v = nullptr;
        if (!gptoss_get_kv(kv.cache, il, cache_k, cache_v)) {
            return -1;
        }

        const uint8_t * qkv_weight    = static_cast<const uint8_t *>(layer.wq->data);
        const uint8_t * output_weight = static_cast<const uint8_t *>(layer.wo->data);
        const uint8_t * moe           = static_cast<const uint8_t *>(layer.ffn_gate_exps->data);

        if (!gptoss_hip_ok(
                gptoss_rms_norm_launch(buffers.cur, static_cast<const float *>(layer.attn_norm->data), buffers.norm,
                                       hparams.f_norm_rms_eps, ubatch.n_tokens, state->stream),
                "attention RMS norm")) {
            return -1;
        }

        if (!gptoss_hip_ok(
                gptoss_q8_qkv_launch(state->q8_qkv, buffers.qkv, buffers.norm,
                                     reinterpret_cast<const int8_t *>(qkv_weight),
                                     reinterpret_cast<const __half *>(qkv_weight + gptoss_qkv_values_size),
                                     static_cast<const float *>(layer.wq_b->data), ubatch.n_tokens, state->stream),
                "QKV projection") ||
            !gptoss_hip_ok(gptoss_qkv_rope_cache_launch(
                               buffers.q, cache_k, cache_v, buffers.qkv, is_swa ? buffers.rope_swa : buffers.rope_base,
                               is_swa ? buffers.swa_write_rows : buffers.base_write_rows, ubatch.n_tokens,
                               state->stream),
                           "QKV RoPE/cache write")) {
            return -1;
        }

        const auto & fa_layout = is_swa ? swa_fa : base_fa;
        if (!gptoss_hip_ok(
                gptoss_fa_launch(is_swa ? state->fa_swa : state->fa_full, buffers.qkv, buffers.q, cache_k, cache_v,
                                 static_cast<const float *>(layer.attn_sinks->data),
                                 is_swa ? buffers.swa_block_table : buffers.base_block_table,
                                 is_swa ? buffers.swa_seq_lens : buffers.base_seq_lens, fa_layout.block_table_stride,
                                 buffers.cu_seqlens_q, static_cast<uint32_t>(spans.size()), ubatch.n_tokens,
                                 state->stream),
                "flash attention")) {
            return -1;
        }

        if (!gptoss_hip_ok(
                gptoss_q8_attention_output_launch(
                    state->q8_attn_out, buffers.next, buffers.qkv, reinterpret_cast<const int8_t *>(output_weight),
                    reinterpret_cast<const __half *>(output_weight + gptoss_attention_output_values_size),
                    static_cast<const float *>(layer.wo_b->data), buffers.cur, ubatch.n_tokens, state->stream),
                "attention output projection") ||
            !gptoss_hip_ok(gptoss_post_attention_rms_norm_launch(
                               buffers.next, static_cast<const float *>(layer.attn_post_norm->data), buffers.cur,
                               buffers.norm, hparams.f_norm_rms_eps, ubatch.n_tokens, state->stream),
                           "post-attention RMS norm")) {
            return -1;
        }
        if (hipblasSgemm(state->hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, gptoss_expert_count, ubatch.n_tokens,
                         gptoss_hidden_size, &alpha, static_cast<const float *>(layer.ffn_gate_inp->data),
                         gptoss_hidden_size, buffers.cur, gptoss_hidden_size, &beta, buffers.router_logits,
                         gptoss_expert_count) != HIPBLAS_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: router projection failed\n", __func__);
            return -1;
        }

        if (!gptoss_hip_ok(gptoss_biased_topk_softmax_launch(
                               buffers.router_logits, static_cast<const float *>(layer.ffn_gate_inp_b->data),
                               buffers.selected_ids, buffers.selected_weights, ubatch.n_tokens, state->stream),
                           "expert selection") ||
            !gptoss_hip_ok(
                gptoss_ogs_build_routes_launch(buffers.selected_ids, buffers.gather_indices, buffers.scatter_indices,
                                               buffers.expert_counts, buffers.route_offsets, buffers.block_offsets,
                                               buffers.block_schedule, route_count, schedule_capacity, state->stream),
                "expert route build")) {
            return -1;
        }

        if (!gptoss_hip_ok(
                gptoss_ogs_w13_launch(
                    state->ogs_w13, buffers.expert_activations, buffers.norm,
                    moe + gptoss_moe_gate_up_values_offset, moe + gptoss_moe_gate_up_scales_offset,
                    static_cast<const float *>(layer.ffn_down_exps_b->data), buffers.gather_indices,
                    buffers.expert_counts, buffers.route_offsets, buffers.block_offsets, buffers.block_schedule,
                    schedule_capacity, state->stream),
                "MoE gate/up projection")) {
            return -1;
        }

        if (!gptoss_hip_ok(
                gptoss_ogs_w2_launch(
                    state->ogs_w2, buffers.expert_outputs, buffers.expert_activations, moe,
                    moe + gptoss_moe_down_scales_offset, static_cast<const float *>(layer.ffn_gate_exps_b->data),
                    buffers.scatter_indices, route_count, buffers.expert_counts, buffers.route_offsets,
                    buffers.block_offsets, buffers.block_schedule, schedule_capacity, state->stream),
                "MoE down projection") ||
            !gptoss_hip_ok(gptoss_moe_combine_launch(buffers.cur, buffers.next, buffers.expert_outputs,
                                                     buffers.selected_weights, ubatch.n_tokens, state->stream),
                           "MoE combine")) {
            return -1;
        }
    }

    size_t output_idx = 0;
    for (uint32_t row = 0; row < ubatch.n_tokens; ++row) {
        if (ubatch.output[row] == 0) {
            continue;
        }
        if (!gptoss_hip_ok(gptoss_rms_norm_launch(
                               buffers.cur + static_cast<size_t>(row) * gptoss_hidden_size,
                               static_cast<const float *>(model.output_norm->data), buffers.norm,
                               hparams.f_norm_rms_eps, 1, state->stream),
                           "output RMS norm") ||
            !gptoss_hip_ok(gptoss_lm_head_mmvq_launch(static_cast<const uint8_t *>(model.output->data),
                                                      buffers.norm, buffers.logits, state->stream),
                           "LM head") ||
            !gptoss_hip_ok(hipMemcpyAsync(logits_out + output_idx * gptoss_vocabulary_size, buffers.logits,
                                          static_cast<size_t>(gptoss_vocabulary_size) * sizeof(float),
                                          hipMemcpyDeviceToHost, state->stream),
                           "logits download")) {
            return -1;
        }
        ++output_idx;
    }

    if (!gptoss_hip_ok(stream_guard.synchronize(), "prefill synchronize")) {
        return -1;
    }
    return 0;
}

int gptoss_decode(llama_context *                ctx,
                  const llama_ubatch &           ubatch,
                  const llama_memory_context_i * mctx,
                  float *                        logits_out) {
    auto * state = static_cast<gptoss_context_state *>(ctx->execution_extension_state);
    if (state == nullptr || ubatch.n_tokens != 1 || ubatch.token == nullptr || mctx == nullptr) {
        return -1;
    }
    std::vector<gptoss_sequence_span> spans;
    if (!gptoss_sequence_spans(ubatch, spans)) {
        return -1;
    }

    const auto *     iswa = static_cast<const llama_kv_cache_iswa_context *>(mctx);
    gptoss_kv_layout base;
    gptoss_kv_layout swa;
    if (!gptoss_build_kv_layout(ubatch, spans, iswa->get_base(), false, base) ||
        !gptoss_build_kv_layout(ubatch, spans, iswa->get_swa(), true, swa)) {
        LLAMA_LOG_ERROR("%s: unsupported KV-cache layout\n", __func__);
        return -1;
    }

    const bool output = ubatch.output[0] != 0;
    if ((logits_out != nullptr) != output) {
        return -1;
    }

    llama_hip_workspace_cursor measure;
    (void) gptoss_make_decode_buffers(measure, base.read_rows.size(), swa.read_rows.size(), output);
    if (!measure.valid()) {
        LLAMA_LOG_ERROR("%s: invalid workspace layout\n", __func__);
        return -1;
    }
    if (!gptoss_hip_ok(state->workspace.reserve(measure.size()), "workspace reserve")) {
        return -1;
    }
    llama_hip_workspace_cursor bind(state->workspace.data(), measure.size());
    auto buffers = gptoss_make_decode_buffers(bind, base.read_rows.size(), swa.read_rows.size(), output);
    if (!bind.valid() || bind.size() != measure.size()) {
        LLAMA_LOG_ERROR("%s: invalid workspace layout\n", __func__);
        return -1;
    }

    gptoss_stream_guard stream_guard(state->stream);
    if (!gptoss_hip_ok(
            hipMemcpyAsync(buffers.token, ubatch.token, sizeof(int32_t), hipMemcpyHostToDevice, state->stream),
                       "token upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.base_rows, base.read_rows.data(),
                                      base.read_rows.size() * sizeof(int32_t), hipMemcpyHostToDevice, state->stream),
                       "base read-row upload") ||
        !gptoss_hip_ok(hipMemcpyAsync(buffers.swa_rows, swa.read_rows.data(),
                                      swa.read_rows.size() * sizeof(int32_t), hipMemcpyHostToDevice, state->stream),
                       "SWA read-row upload")) {
        return -1;
    }

    const auto & model   = ctx->get_model();
    const auto & hparams = model.hparams;
    const auto & cparams = ctx->get_cparams();

    if (!gptoss_hip_ok(
            gptoss_embedding_q8_0_launch(buffers.cur, static_cast<const uint8_t *>(model.tok_embd->data),
                                         buffers.token, 1, state->stream),
            "token embedding")) {
        return -1;
    }

    float * current = buffers.cur;
    float * next    = buffers.next;

    for (uint32_t il = 0; il < gptoss_layer_count; ++il) {
        const llama_layer & layer  = model.layers[il];
        const bool          is_swa = hparams.is_swa(il);
        const auto &        kv     = is_swa ? swa : base;

        __half * cache_k = nullptr;
        __half * cache_v = nullptr;
        if (!gptoss_get_kv(kv.cache, il, cache_k, cache_v)) {
            return -1;
        }

        const float freq_base  = model.get_rope_freq_base(cparams, il);
        const float freq_scale = model.get_rope_freq_scale(cparams, il);
        float       corr[2];
        ggml_rope_yarn_corr_dims(gptoss_head_size, cparams.n_ctx_orig_yarn, freq_base, cparams.yarn_beta_fast,
                                 cparams.yarn_beta_slow, corr);

        const uint8_t * qkv_weight    = static_cast<const uint8_t *>(layer.wq->data);
        const uint8_t * output_weight = static_cast<const uint8_t *>(layer.wo->data);
        const uint8_t * moe           = static_cast<const uint8_t *>(layer.ffn_gate_exps->data);

        gptoss_decode_layer_params params = {};
        params.next                      = next;
        params.cur                       = current;
        params.rms_partials              = buffers.rms_partials;
        params.activation_scratch        = buffers.activation_scratch;
        params.query                     = buffers.query;
        params.attn_parts                = buffers.attention_parts;
        params.attn_meta                 = buffers.attention_meta;
        params.router                    = buffers.router_scores;
        params.expert_ids                = buffers.selected_experts;
        params.expert_weights            = buffers.selected_weights;
        params.cache_k                   = cache_k;
        params.cache_v                   = cache_v;
        params.kv_rows                   = is_swa ? buffers.swa_rows : buffers.base_rows;
        params.attn_norm                 = static_cast<const float *>(layer.attn_norm->data);
        params.qkv_values                = reinterpret_cast<const int8_t *>(qkv_weight);
        params.attn_q_bias               = static_cast<const float *>(layer.wq_b->data);
        params.attn_k_bias               = static_cast<const float *>(layer.wk_b->data);
        params.attn_v_bias               = static_cast<const float *>(layer.wv_b->data);
        params.attn_output_values        = reinterpret_cast<const int8_t *>(output_weight);
        params.attn_output_bias    = static_cast<const float *>(layer.wo_b->data);
        params.attn_sinks          = static_cast<const float *>(layer.attn_sinks->data);
        params.post_attention_norm = static_cast<const float *>(layer.attn_post_norm->data);
        params.router_weight       = static_cast<const float *>(layer.ffn_gate_inp->data);
        params.router_bias         = static_cast<const float *>(layer.ffn_gate_inp_b->data);
        params.moe_down_values     = moe;
        params.moe_gate_up_values  = moe + gptoss_moe_gate_up_values_offset;
        params.moe_down_bias       = static_cast<const float *>(layer.ffn_gate_exps_b->data);
        params.moe_gate_up_bias    = static_cast<const float *>(layer.ffn_down_exps_b->data);
        params.n_kv                = static_cast<uint32_t>(kv.read_rows.size());
        params.kv_write_row        = static_cast<uint32_t>(kv.write_rows[0]);
        params.attn_parallel_blocks = buffers.attention_partitions;
        params.position             = ubatch.pos[0];
        params.rms_epsilon          = hparams.f_norm_rms_eps;
        params.rope_freq_scale      = freq_scale;
        params.rope_ext_factor      = cparams.yarn_ext_factor;
        params.rope_attn_factor     = cparams.yarn_attn_factor;
        params.rope_corr_low        = corr[0];
        params.rope_corr_high       = corr[1];
        params.rope_theta_scale     = std::pow(freq_base, -2.0f / gptoss_head_size);
        params.reuse_attention_rms  = il != 0;

        if (!gptoss_hip_ok(gptoss_decode_layer_launch(is_swa, params, state->stream), "decode layer")) {
            return -1;
        }

        std::swap(current, next);
    }

    if (output) {
        if (!gptoss_hip_ok(gptoss_rms_norm_launch(
                               current, static_cast<const float *>(model.output_norm->data), buffers.activation_scratch,
                               hparams.f_norm_rms_eps, 1, state->stream),
                           "output RMS norm") ||
            !gptoss_hip_ok(gptoss_lm_head_mmvq_launch(static_cast<const uint8_t *>(model.output->data),
                                                      buffers.activation_scratch, buffers.logits, state->stream),
                           "LM head") ||
            !gptoss_hip_ok(
                               hipMemcpyAsync(logits_out, buffers.logits, static_cast<size_t>(gptoss_vocabulary_size) * sizeof(float),
                               hipMemcpyDeviceToHost, state->stream),
                "logits download")) {
            return -1;
        }
    }

    return gptoss_hip_ok(stream_guard.synchronize(), "decode synchronize") ? 0 : -1;
}

size_t gptoss_tensor_alloc_size(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_MXFP4 || std::strstr(tensor->name, ".ffn_up_exps.weight") == nullptr) {
        return 0;
    }

    const size_t hidden       = static_cast<size_t>(tensor->ne[0]);
    const size_t intermediate = static_cast<size_t>(tensor->ne[1]);
    const size_t n_experts    = static_cast<size_t>(tensor->ne[2]);

    const size_t padded_hidden       = GGML_PAD(hidden, gptoss_ogs_alignment);
    const size_t padded_intermediate = GGML_PAD(intermediate, gptoss_ogs_alignment);

    const size_t packed_gate_size = n_experts * padded_intermediate * ggml_row_size(tensor->type, padded_hidden);
    const size_t packed_up_size   = packed_gate_size;
    const size_t packed_down_size = n_experts * padded_hidden * ggml_row_size(tensor->type, padded_intermediate);

    const size_t native_gate_size = ggml_nbytes(tensor);
    const size_t native_down_size = ggml_nbytes(tensor);

    return packed_gate_size + packed_up_size + packed_down_size - native_gate_size - native_down_size;
}

}  // namespace

const llama_execution_extension gptoss_execution_extension = {
    gptoss_model_init, gptoss_context_init, gptoss_context_free,
    gptoss_prefill,    gptoss_decode,       gptoss_tensor_alloc_size,
};
