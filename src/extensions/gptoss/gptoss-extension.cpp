#include "gptoss-extension.h"

#include "extensions/llama-execution-extension.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
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

constexpr uint32_t gptoss_layer_count           = 24;
constexpr uint32_t gptoss_hidden_size           = 2880;
constexpr uint32_t gptoss_intermediate_size     = 2880;
constexpr uint32_t gptoss_query_size            = 4096;
constexpr uint32_t gptoss_key_value_size        = 512;
constexpr uint32_t gptoss_qkv_size              = gptoss_query_size + 2 * gptoss_key_value_size;
constexpr uint32_t gptoss_head_size             = 64;
constexpr uint32_t gptoss_query_head_count      = 64;
constexpr uint32_t gptoss_kv_head_count         = 8;
constexpr uint32_t gptoss_expert_count          = 32;
constexpr uint32_t gptoss_expert_used_count     = 4;
constexpr uint32_t gptoss_vocabulary_size       = 201088;
constexpr uint32_t gptoss_swa_size              = 128;
constexpr uint32_t gptoss_max_attention_parts   = 12;
constexpr uint32_t gptoss_quant_block_size      = 32;
constexpr uint32_t gptoss_ogs_block_m           = 64;
constexpr uint32_t gptoss_q8_shared_memory      = 16384;
constexpr uint32_t gptoss_fa_tile_size          = 32;
constexpr uint32_t gptoss_fa_block_q            = 8;
constexpr uint32_t gptoss_fa_threads            = 128;
constexpr uint32_t gptoss_fa_shared_memory      = 8192;
constexpr uint32_t gptoss_ogs_shared_memory     = 16384;

constexpr size_t gptoss_qkv_values_size              = static_cast<size_t>(gptoss_qkv_size) * gptoss_hidden_size;
constexpr size_t gptoss_attention_output_values_size = static_cast<size_t>(gptoss_hidden_size) * gptoss_query_size;

constexpr size_t gptoss_moe_down_scales_offset    = 150994944;
constexpr size_t gptoss_moe_gate_up_values_offset = 160432128;
constexpr size_t gptoss_moe_gate_up_scales_offset = 462422016;

constexpr size_t gptoss_q8_1_row_size = static_cast<size_t>(gptoss_hidden_size / gptoss_quant_block_size) * 36;

constexpr const char * gptoss_fa_name = "kernel_unified_attention_2d";

struct gptoss_context_state {
    int device = 0;
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

    const uint8_t * token_embedding      = nullptr;
    void *          token_embedding_copy = nullptr;

    void * workspace      = nullptr;
    size_t workspace_size = 0;
};

struct gptoss_q8_qkv_args {
    void *       output;
    const void * activation;
    const void * values;
    const void * scales;
    const void * bias;
    int32_t      n_tokens;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_q8_attention_output_args {
    void *       output;
    const void * activation;
    const void * values;
    const void * scales;
    const void * bias;
    const void * residual;
    int32_t      n_tokens;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_fa_args {
    void *       output;
    const void * query;
    const void * key_cache;
    const void * value_cache;
    const void * sinks;
    const void * block_table;
    const void * seq_lens;
    int64_t      block_table_stride;
    const void * cu_seqlens_q;
    int32_t      n_sequences;
    uint32_t     padding;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_ogs_w13_args {
    void *       output;
    void *       output_ptr;
    const void * activation;
    const void * activation_ptr;
    const void * values;
    const void * values_ptr;
    const void * scales;
    const void * bias;
    const void * gather_indices;
    const void * expert_counts;
    const void * route_offsets;
    const void * block_offsets;
    const void * block_schedule;
    int32_t      grid_m;
    uint32_t     padding;
    void *       global_scratch;
    void *       profile_scratch;
};

struct gptoss_ogs_w2_args {
    void *       output;
    void *       output_ptr;
    const void * activation;
    const void * activation_ptr;
    const void * values;
    const void * values_ptr;
    const void * scales;
    const void * bias;
    const void * scatter_indices;
    int32_t      route_count;
    uint32_t     padding_0;
    const void * expert_counts;
    const void * route_offsets;
    const void * block_offsets;
    const void * block_schedule;
    int32_t      grid_m;
    uint32_t     padding_1;
    void *       global_scratch;
    void *       profile_scratch;
};

static_assert(sizeof(gptoss_fa_args) == 96);
static_assert(sizeof(gptoss_ogs_w13_args) == 128);
static_assert(sizeof(gptoss_ogs_w2_args) == 136);

template <typename T>
hipError_t gptoss_aot_launch(hipFunction_t function,
                             uint32_t      blocks,
                             uint32_t      threads,
                             uint32_t      shared_memory,
                             T &           args,
                             hipStream_t   stream) {
    size_t args_size = sizeof(args);
    void * config[]  = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size, HIP_LAUNCH_PARAM_END,
    };

    return hipModuleLaunchKernel(function, blocks, 1, 1, threads, 1, 1, shared_memory, stream, nullptr, config);
}

hipError_t gptoss_fa_launch(hipFunction_t function,
                            uint32_t      query_blocks,
                            gptoss_fa_args & args,
                            hipStream_t   stream) {
    size_t args_size = sizeof(args);
    void * config[]  = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size, HIP_LAUNCH_PARAM_END,
    };

    return hipModuleLaunchKernel(
        function, gptoss_kv_head_count, query_blocks, 1, gptoss_fa_threads, 1, 1, gptoss_fa_shared_memory, stream,
        nullptr, config);
}

hipError_t gptoss_q8_qkv_launch(
    hipFunction_t function, uint32_t blocks, gptoss_q8_qkv_args & args, hipStream_t stream) {
    void * params[] = {
        &args.output, &args.activation, &args.values,         &args.scales,
        &args.bias,   &args.n_tokens,   &args.global_scratch, &args.profile_scratch,
    };
    return hipModuleLaunchKernel(function, blocks, 1, 1, 128, 1, 1, gptoss_q8_shared_memory, stream, params, nullptr);
}

hipError_t gptoss_q8_attention_output_launch(hipFunction_t                     function,
                                             uint32_t                          blocks,
                                             gptoss_q8_attention_output_args & args,
                                             hipStream_t                       stream) {
    void * params[] = {
        &args.output,   &args.activation, &args.values,         &args.scales,          &args.bias,
        &args.residual, &args.n_tokens,   &args.global_scratch, &args.profile_scratch,
    };
    return hipModuleLaunchKernel(function, blocks, 1, 1, 128, 1, 1, gptoss_q8_shared_memory, stream, params, nullptr);
}

bool gptoss_hip_ok(hipError_t error, const char * operation) {
    if (error == hipSuccess) {
        return true;
    }

    LLAMA_LOG_ERROR("gptoss: %s failed: %s\n", operation, hipGetErrorString(error));
    return false;
}

int gptoss_device(const llama_model & model) {
    const auto * device = model.dev_layer(0);
    auto *       reg    = ggml_backend_cuda_reg();

    for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == device) {
            return i;
        }
    }

    return -1;
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

bool gptoss_workspace_reserve(gptoss_context_state * state, size_t size) {
    if (size <= state->workspace_size) {
        return true;
    }

    if (state->workspace != nullptr) {
        if (!gptoss_hip_ok(hipStreamSynchronize(state->stream), "workspace synchronize") ||
            !gptoss_hip_ok(hipFree(state->workspace), "workspace free")) {
            return false;
        }
        state->workspace      = nullptr;
        state->workspace_size = 0;
    }

    if (!gptoss_hip_ok(hipMalloc(&state->workspace, size), "workspace allocation")) {
        return false;
    }

    state->workspace_size = size;
    return true;
}

struct gptoss_arena {
    explicit gptoss_arena(void * data) : data(static_cast<uint8_t *>(data)) {}

    template <typename T> T * take(size_t count) {
        offset     = GGML_PAD(offset, 256);
        T * result = data == nullptr ? nullptr : reinterpret_cast<T *>(data + offset);
        offset += count * sizeof(T);
        return result;
    }

    uint8_t * data;
    size_t    offset = 0;
};

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

    if (cache == nullptr || slot.empty() || slot.n_stream() * slot.size() != ubatch.n_tokens) {
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
        const llama_pos first_visible  = sliding_window ? first_position - (gptoss_swa_size - 1) : 0;
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

    uint8_t * final_q8;
    float *   logits;

    size_t size;
};

gptoss_prefill_buffers gptoss_make_prefill_buffers(void *   workspace,
                                                   uint32_t n_tokens,
                                                   bool     output,
                                                   uint32_t n_sequences,
                                                   uint32_t n_base_blocks,
                                                   uint32_t n_swa_blocks) {
    const uint32_t route_count = n_tokens * gptoss_expert_used_count;
    const uint32_t schedule_capacity =
        (route_count + gptoss_ogs_block_m - 1) / gptoss_ogs_block_m + gptoss_expert_count - 1;
    gptoss_arena           arena(workspace);
    gptoss_prefill_buffers result = {};

    result.tokens             = arena.take<int32_t>(n_tokens);
    result.positions          = arena.take<int32_t>(n_tokens);
    result.base_write_rows    = arena.take<int64_t>(n_tokens);
    result.swa_write_rows     = arena.take<int64_t>(n_tokens);
    result.base_block_table   = arena.take<int32_t>(n_base_blocks);
    result.swa_block_table    = arena.take<int32_t>(n_swa_blocks);
    result.cu_seqlens_q       = arena.take<int32_t>(n_sequences + 1);
    result.base_seq_lens      = arena.take<int32_t>(n_sequences);
    result.swa_seq_lens       = arena.take<int32_t>(n_sequences);
    result.cur                = arena.take<float>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.next               = arena.take<float>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.norm               = arena.take<__half>(static_cast<size_t>(n_tokens) * gptoss_hidden_size);
    result.rope_base          = arena.take<float>(static_cast<size_t>(n_tokens) * gptoss_head_size);
    result.rope_swa           = arena.take<float>(static_cast<size_t>(n_tokens) * gptoss_head_size);
    result.qkv                = arena.take<__half>(static_cast<size_t>(n_tokens) * gptoss_qkv_size);
    result.q                  = arena.take<__half>(static_cast<size_t>(n_tokens) * gptoss_query_size);
    result.router_logits      = arena.take<float>(static_cast<size_t>(n_tokens) * gptoss_expert_count);
    result.selected_ids       = arena.take<int32_t>(route_count);
    result.selected_weights   = arena.take<float>(route_count);
    result.gather_indices     = arena.take<int32_t>(route_count);
    result.scatter_indices    = arena.take<int32_t>(route_count);
    result.expert_counts      = arena.take<int32_t>(gptoss_expert_count);
    result.route_offsets      = arena.take<int32_t>(gptoss_expert_count + 1);
    result.block_offsets      = arena.take<int32_t>(gptoss_expert_count + 1);
    result.block_schedule     = arena.take<int32_t>(schedule_capacity);
    result.expert_activations = arena.take<__half>(static_cast<size_t>(route_count) * gptoss_intermediate_size);
    result.expert_outputs     = arena.take<float>(static_cast<size_t>(route_count) * gptoss_hidden_size);
    result.final_q8           = arena.take<uint8_t>(output ? gptoss_q8_1_row_size : 0);
    result.logits             = arena.take<float>(output ? gptoss_vocabulary_size : 0);
    result.size               = arena.offset;

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

    uint8_t * final_q8;
    float *   logits;

    size_t size;
};

gptoss_decode_buffers gptoss_make_decode_buffers(void *   workspace,
                                                 size_t   n_base_rows,
                                                 size_t   n_swa_rows,
                                                 bool     output) {
    const uint32_t partitions = static_cast<uint32_t>(std::min<size_t>(
        gptoss_max_attention_parts,
        std::max<size_t>(2, (n_base_rows + gptoss_swa_size - 1) / gptoss_swa_size)));

    gptoss_arena          arena(workspace);
    gptoss_decode_buffers result = {};

    result.token            = arena.take<int32_t>(1);
    result.base_rows        = arena.take<int32_t>(n_base_rows + n_swa_rows);
    result.swa_rows         = result.base_rows == nullptr ? nullptr : result.base_rows + n_base_rows;
    result.cur              = arena.take<float>(gptoss_hidden_size);
    result.next             = arena.take<float>(gptoss_hidden_size);
    result.rms_partials     = arena.take<float>(gptoss_decode_grid_blocks);
    result.activation_scratch =
        arena.take<__half>(gptoss_hidden_size * (1 + gptoss_expert_used_count));
    result.query            = arena.take<__half>(gptoss_query_size);
    result.router_scores    = arena.take<float>(gptoss_expert_count);
    result.selected_experts = arena.take<int32_t>(gptoss_expert_used_count);
    result.selected_weights = arena.take<float>(gptoss_expert_used_count);
    result.attention_parts =
        arena.take<float>(static_cast<size_t>(gptoss_query_head_count) * partitions * gptoss_head_size);
    result.attention_meta       = arena.take<float2>(static_cast<size_t>(gptoss_query_head_count) * partitions);
    result.attention_partitions = partitions;
    result.final_q8             = arena.take<uint8_t>(output ? gptoss_q8_1_row_size : 0);
    result.logits               = arena.take<float>(output ? gptoss_vocabulary_size : 0);
    result.size                 = arena.offset;

    return result;
}

void gptoss_context_free(llama_context * ctx) {
    auto * state = static_cast<gptoss_context_state *>(ctx->execution_extension_state);

    if (state == nullptr) {
        return;
    }

    (void) hipSetDevice(state->device);
    if (state->stream != nullptr) {
        (void) hipStreamSynchronize(state->stream);
    }

    if (state->hipblas != nullptr) {
        (void) hipblasDestroy(state->hipblas);
    }
    if (state->token_embedding_copy != nullptr) {
        (void) hipFree(state->token_embedding_copy);
    }
    if (state->workspace != nullptr) {
        (void) hipFree(state->workspace);
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

    const int       device = gptoss_device(*model);
    hipDeviceProp_t properties{};
    if (device < 0 || hipSetDevice(device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess ||
        std::strncmp(properties.gcnArchName, "gfx1201", 7) != 0 || properties.warpSize != 32 ||
        !properties.cooperativeLaunch) {
        LLAMA_LOG_ERROR("%s: unsupported device\n", __func__);
        return false;
    }

    const auto tensor_is = [](const ggml_tensor * tensor, ggml_type type) {
        return tensor != nullptr && tensor->type == type && tensor->data != nullptr && tensor->buffer != nullptr;
    };
    const auto device_tensor_is = [model, &tensor_is](const ggml_tensor * tensor, ggml_type type) {
        return tensor_is(tensor, type) &&
               ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) == model->dev_layer(0);
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

    if (!tensor_is(model->tok_embd, GGML_TYPE_Q8_0) || !device_tensor_is(model->output, GGML_TYPE_Q8_0) ||
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
    const auto & model   = ctx->get_model();
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

    const int device = gptoss_device(model);
    if (device < 0 || hipSetDevice(device) != hipSuccess) {
        return false;
    }

    auto * state                   = new gptoss_context_state;
    ctx->execution_extension_state = state;
    state->device                  = device;

    if (hipStreamCreateWithFlags(&state->stream, hipStreamNonBlocking) != hipSuccess) {
        gptoss_context_free(ctx);
        return false;
    }

    state->token_embedding      = static_cast<const uint8_t *>(model.tok_embd->data);
    auto * token_embedding_buft = ggml_backend_buffer_get_type(model.tok_embd->buffer);
    auto * token_embedding_dev  = ggml_backend_buft_get_device(token_embedding_buft);
    if (token_embedding_dev == nullptr || ggml_backend_dev_type(token_embedding_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        const size_t size = ggml_nbytes(model.tok_embd);
        if (hipMalloc(&state->token_embedding_copy, size) != hipSuccess ||
            hipMemcpy(state->token_embedding_copy, model.tok_embd->data, size, hipMemcpyHostToDevice) != hipSuccess) {
            gptoss_context_free(ctx);
            return false;
        }
        state->token_embedding = static_cast<const uint8_t *>(state->token_embedding_copy);
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
    if (hipSetDevice(state->device) != hipSuccess) {
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

    std::vector<int32_t> output_rows;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i] != 0) {
            output_rows.push_back(static_cast<int32_t>(i));
        }
    }
    if ((logits_out != nullptr) != !output_rows.empty()) {
        return -1;
    }

    auto buffers = gptoss_make_prefill_buffers(
        nullptr, ubatch.n_tokens, !output_rows.empty(), static_cast<uint32_t>(spans.size()),
        static_cast<uint32_t>(base_fa.block_table.size()), static_cast<uint32_t>(swa_fa.block_table.size()));
    if (!gptoss_workspace_reserve(state, buffers.size)) {
        return -1;
    }
    buffers =
        gptoss_make_prefill_buffers(state->workspace, ubatch.n_tokens, !output_rows.empty(),
                                    static_cast<uint32_t>(spans.size()),
                                    static_cast<uint32_t>(base_fa.block_table.size()),
                                    static_cast<uint32_t>(swa_fa.block_table.size()));

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
                buffers.cur, state->token_embedding, buffers.tokens, ubatch.n_tokens, state->stream),
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
    const uint32_t fa_query_blocks =
        ubatch.n_tokens / gptoss_fa_block_q + static_cast<uint32_t>(spans.size());
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
                gptoss_attention_rms_norm_launch(buffers.cur, static_cast<const float *>(layer.attn_norm->data),
                                                 buffers.norm, hparams.f_norm_rms_eps, ubatch.n_tokens, state->stream),
                "attention RMS norm")) {
            return -1;
        }

        gptoss_q8_qkv_args qkv_args = {
            buffers.qkv,      buffers.norm,
            qkv_weight,       qkv_weight + gptoss_qkv_values_size,
            layer.wq_b->data, static_cast<int32_t>(ubatch.n_tokens),
            nullptr,          nullptr,
        };
        const uint32_t qkv_grid = ((ubatch.n_tokens + 63) / 64) * 40;
        if (!gptoss_hip_ok(gptoss_q8_qkv_launch(state->q8_qkv, qkv_grid, qkv_args, state->stream),
                           "QKV projection") ||
            !gptoss_hip_ok(gptoss_qkv_rope_cache_launch(
                               buffers.q, cache_k, cache_v, buffers.qkv, is_swa ? buffers.rope_swa : buffers.rope_base,
                               is_swa ? buffers.swa_write_rows : buffers.base_write_rows, ubatch.n_tokens,
                               state->stream),
                           "QKV RoPE/cache write")) {
            return -1;
        }

        const auto & fa_layout = is_swa ? swa_fa : base_fa;
        gptoss_fa_args fa_args = {
            buffers.qkv,
            buffers.q,
            cache_k,
            cache_v,
            layer.attn_sinks->data,
            is_swa ? buffers.swa_block_table : buffers.base_block_table,
            is_swa ? buffers.swa_seq_lens : buffers.base_seq_lens,
            fa_layout.block_table_stride,
            buffers.cu_seqlens_q,
            static_cast<int32_t>(spans.size()),
            0,
            nullptr,
            nullptr,
        };
        if (!gptoss_hip_ok(
                gptoss_fa_launch(is_swa ? state->fa_swa : state->fa_full, fa_query_blocks, fa_args, state->stream),
                "flash attention")) {
            return -1;
        }

        gptoss_q8_attention_output_args output_args = {
            buffers.next,
            buffers.qkv,
            output_weight,
            output_weight + gptoss_attention_output_values_size,
            layer.wo_b->data,
            buffers.cur,
            static_cast<int32_t>(ubatch.n_tokens),
            nullptr,
            nullptr,
        };
        const uint32_t output_grid = ((ubatch.n_tokens + 63) / 64) * 23;
        if (!gptoss_hip_ok(
                gptoss_q8_attention_output_launch(state->q8_attn_out, output_grid, output_args, state->stream),
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

        gptoss_ogs_w13_args w13_args = {
            buffers.expert_activations,
            buffers.expert_activations,
            buffers.norm,
            buffers.norm,
            moe + gptoss_moe_gate_up_values_offset,
            moe + gptoss_moe_gate_up_values_offset,
            moe + gptoss_moe_gate_up_scales_offset,
            layer.ffn_down_exps_b->data,
            buffers.gather_indices,
            buffers.expert_counts,
            buffers.route_offsets,
            buffers.block_offsets,
            buffers.block_schedule,
            static_cast<int32_t>(schedule_capacity),
            0,
            nullptr,
            nullptr,
        };
        if (!gptoss_hip_ok(
                gptoss_aot_launch(state->ogs_w13, schedule_capacity * 45, 128, gptoss_ogs_shared_memory, w13_args,
                                  state->stream),
                "MoE gate/up projection")) {
            return -1;
        }

        gptoss_ogs_w2_args w2_args = {
            buffers.expert_outputs,
            buffers.expert_outputs,
            buffers.expert_activations,
            buffers.expert_activations,
            moe,
            moe,
            moe + gptoss_moe_down_scales_offset,
            layer.ffn_gate_exps_b->data,
            buffers.scatter_indices,
            static_cast<int32_t>(route_count),
            0,
            buffers.expert_counts,
            buffers.route_offsets,
            buffers.block_offsets,
            buffers.block_schedule,
            static_cast<int32_t>(schedule_capacity),
            0,
            nullptr,
            nullptr,
        };
        if (!gptoss_hip_ok(
                gptoss_aot_launch(state->ogs_w2, schedule_capacity * 23, 128, gptoss_ogs_shared_memory, w2_args,
                                  state->stream),
                "MoE down projection") ||
            !gptoss_hip_ok(gptoss_moe_combine_launch(buffers.cur, buffers.next, buffers.expert_outputs,
                                                     buffers.selected_weights, ubatch.n_tokens, state->stream),
                           "MoE combine")) {
            return -1;
        }
    }

    for (size_t i = 0; i < output_rows.size(); ++i) {
        if (!gptoss_hip_ok(gptoss_output_rms_norm_quantize_launch(
                               buffers.cur, static_cast<const float *>(model.output_norm->data),
                               output_rows[i], buffers.final_q8, hparams.f_norm_rms_eps, state->stream),
                           "output RMS norm") ||
            !gptoss_hip_ok(gptoss_lm_head_mmvq_launch(static_cast<const uint8_t *>(model.output->data),
                                                      buffers.final_q8, buffers.logits, state->stream),
                           "LM head") ||
            !gptoss_hip_ok(hipMemcpyAsync(logits_out + i * gptoss_vocabulary_size, buffers.logits,
                                          static_cast<size_t>(gptoss_vocabulary_size) * sizeof(float),
                                          hipMemcpyDeviceToHost, state->stream),
                           "logits download")) {
            return -1;
        }
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
    if (hipSetDevice(state->device) != hipSuccess) {
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

    auto buffers = gptoss_make_decode_buffers(nullptr, base.read_rows.size(), swa.read_rows.size(), output);
    if (!gptoss_workspace_reserve(state, buffers.size)) {
        return -1;
    }
    buffers = gptoss_make_decode_buffers(state->workspace, base.read_rows.size(), swa.read_rows.size(), output);

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
            gptoss_embedding_q8_0_launch(buffers.cur, state->token_embedding, buffers.token, 1, state->stream),
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
        if (!gptoss_hip_ok(gptoss_output_rms_norm_quantize_launch(
                               current, static_cast<const float *>(model.output_norm->data), 0, buffers.final_q8,
                               hparams.f_norm_rms_eps, state->stream),
                           "output RMS norm") ||
            !gptoss_hip_ok(gptoss_lm_head_mmvq_launch(static_cast<const uint8_t *>(model.output->data),
                                                      buffers.final_q8, buffers.logits, state->stream),
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

}  // namespace

const llama_execution_extension gptoss_execution_extension = {
    gptoss_model_init, gptoss_context_init, gptoss_context_free,
    gptoss_prefill,    gptoss_decode,       gptoss_tensor_alloc_size,
};
