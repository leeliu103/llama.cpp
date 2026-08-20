#include "gptoss-kv.h"

#include "ggml.h"
#include "gptoss-config.h"
#include "gptoss-kernel-aot.h"
#include "llama-batch.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace {

bool build_query_offsets(const llama_ubatch & ubatch, std::vector<int32_t> & offsets) {
    if (ubatch.n_tokens == 0 || ubatch.n_pos != 1 ||
        ubatch.n_tokens > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    offsets.reserve(static_cast<size_t>(std::min(ubatch.n_seqs, ubatch.n_tokens)) + 1);
    offsets.push_back(0);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] <= 0 || ubatch.pos[i] < 0) {
            return false;
        }

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            if (ubatch.seq_id[i][s] < 0) {
                return false;
            }
        }

        if (i == 0) {
            continue;
        }

        const llama_seq_id sequence = ubatch.seq_id[i][0];
        if (ubatch.seq_id[i - 1][0] != sequence) {
            for (const int32_t begin : offsets) {
                if (ubatch.seq_id[begin][0] == sequence) {
                    return false;
                }
            }
            offsets.push_back(static_cast<int32_t>(i));
            continue;
        }

        if (static_cast<int64_t>(ubatch.pos[i]) != static_cast<int64_t>(ubatch.pos[i - 1]) + 1 ||
            ubatch.n_seq_id[i] != ubatch.n_seq_id[i - 1]) {
            return false;
        }
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            if (ubatch.seq_id[i][s] != ubatch.seq_id[i - 1][s]) {
                return false;
            }
        }
    }

    offsets.push_back(static_cast<int32_t>(ubatch.n_tokens));
    return true;
}

bool build_kv_cache_rows(const llama_ubatch &            ubatch,
                         const std::vector<int32_t> &    query_offsets,
                         const llama_kv_cache_context *  context,
                         bool                            sliding_window,
                         gptoss_kv_cache_rows &          result) {
    const auto * cache = context->get_cache();
    const auto & slot  = context->get_slot_info();

    if (cache == nullptr || cache->get_has_shift() || slot.empty() || slot.n_stream() * slot.size() != ubatch.n_tokens) {
        return false;
    }

    const uint32_t cache_size = cache->get_size();
    const uint32_t n_streams  = cache->get_n_stream();
    result.cache              = cache;
    result.write_rows.resize(ubatch.n_tokens);

    for (uint32_t stream = 0; stream < slot.n_stream(); ++stream) {
        for (uint32_t token = 0; token < slot.size(); ++token) {
            const uint32_t i = stream * slot.size() + token;
            if (static_cast<uint32_t>(slot.strm[stream]) >= n_streams || slot.idxs[stream][token] >= cache_size) {
                return false;
            }
            result.write_rows[i] = static_cast<int64_t>(slot.strm[stream]) * cache_size + slot.idxs[stream][token];
        }
    }

    struct cell_row {
        llama_pos position;
        int32_t   row;
    };

    for (size_t sequence_index = 0; sequence_index + 1 < query_offsets.size(); ++sequence_index) {
        const uint32_t begin    = static_cast<uint32_t>(query_offsets[sequence_index]);
        const uint32_t end      = static_cast<uint32_t>(query_offsets[sequence_index + 1]);
        const uint32_t n_tokens = end - begin;
        const llama_seq_id sequence = ubatch.seq_id[begin][0];

        const uint32_t stream = static_cast<uint32_t>(result.write_rows[begin] / cache_size);
        for (uint32_t i = begin + 1; i < end; ++i) {
            if (static_cast<uint32_t>(result.write_rows[i] / cache_size) != stream) {
                return false;
            }
        }

        const llama_pos first_position = ubatch.pos[begin];
        const llama_pos last_position  = ubatch.pos[end - 1];
        const llama_pos first_visible =
            sliding_window ? first_position - static_cast<llama_pos>(gptoss_swa_size - 1) : 0;
        const auto & cells = cache->get_cells(sequence);

        std::vector<cell_row> visible_rows;
        visible_rows.reserve(cells.get_used());
        for (uint32_t row = 0; row < cells.size(); ++row) {
            if (cells.is_empty(row) || !cells.seq_has(row, sequence)) {
                continue;
            }

            const llama_pos position = cells.pos_get(row);
            if (position <= last_position && position >= first_visible) {
                const uint64_t global_row = static_cast<uint64_t>(stream) * cache_size + row;
                if (global_row > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                    return false;
                }
                visible_rows.push_back({ position, static_cast<int32_t>(global_row) });
            }
        }

        std::sort(visible_rows.begin(), visible_rows.end(),
                  [](const cell_row & a, const cell_row & b) { return a.position < b.position; });

        if (visible_rows.size() < n_tokens ||
            visible_rows.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        for (size_t i = 1; i < visible_rows.size(); ++i) {
            if (static_cast<int64_t>(visible_rows[i].position) !=
                static_cast<int64_t>(visible_rows[i - 1].position) + 1) {
                return false;
            }
        }
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const auto & row = visible_rows[visible_rows.size() - n_tokens + i];
            if (row.position != ubatch.pos[begin + i] || row.row != result.write_rows[begin + i]) {
                return false;
            }
        }

        for (const auto & row : visible_rows) {
            result.read_rows.push_back(row.row);
        }
        result.sequence_lengths.push_back(static_cast<int32_t>(visible_rows.size()));
    }

    return !result.read_rows.empty();
}

}  // namespace

bool gptoss_build_kv_batch(const llama_ubatch &                ubatch,
                           const llama_kv_cache_iswa_context & context,
                           gptoss_kv_batch &                   batch) {
    gptoss_kv_batch result;

    if (!build_query_offsets(ubatch, result.query_offsets) ||
        !build_kv_cache_rows(ubatch, result.query_offsets, context.get_base(), false, result.base) ||
        !build_kv_cache_rows(ubatch, result.query_offsets, context.get_swa(), true, result.swa)) {
        return false;
    }

    batch = std::move(result);
    return true;
}

bool gptoss_build_fa_block_table(const gptoss_kv_cache_rows & rows, gptoss_fa_block_table & table) {
    if (rows.sequence_lengths.empty()) {
        return false;
    }

    const size_t n_sequences = rows.sequence_lengths.size();
    size_t       n_rows      = 0;
    int32_t      max_seq_len = 0;

    for (const int32_t sequence_length : rows.sequence_lengths) {
        if (sequence_length <= 0 || static_cast<size_t>(sequence_length) > rows.read_rows.size() - n_rows) {
            return false;
        }
        n_rows += static_cast<size_t>(sequence_length);
        max_seq_len = std::max(max_seq_len, sequence_length);
    }
    if (n_rows != rows.read_rows.size()) {
        return false;
    }

    if (max_seq_len > std::numeric_limits<int32_t>::max() - static_cast<int32_t>(gptoss_fa_tile_size - 1)) {
        return false;
    }

    gptoss_fa_block_table result;
    result.stride = (max_seq_len + gptoss_fa_tile_size - 1) / gptoss_fa_tile_size * gptoss_fa_tile_size;
    const size_t stride = static_cast<size_t>(result.stride);
    if (stride > std::numeric_limits<uint32_t>::max() / n_sequences) {
        return false;
    }
    result.row_indices.resize(stride * n_sequences);

    size_t begin = 0;
    for (size_t i = 0; i < n_sequences; ++i) {
        const size_t end = begin + static_cast<size_t>(rows.sequence_lengths[i]);
        int32_t *    dst = result.row_indices.data() + i * stride;
        std::copy(rows.read_rows.begin() + begin, rows.read_rows.begin() + end, dst);
        std::fill(dst + end - begin, dst + stride, rows.read_rows[end - 1]);
        begin = end;
    }

    table = std::move(result);
    return true;
}

bool gptoss_get_kv_storage(const llama_kv_cache * cache, uint32_t layer, __half *& k, __half *& v) {
    if (cache == nullptr) {
        return false;
    }

    const ggml_tensor * tensor_k = cache->get_k_storage(layer);
    const ggml_tensor * tensor_v = cache->get_v_storage(layer);

    if (tensor_k == nullptr || tensor_v == nullptr || tensor_k->type != GGML_TYPE_F16 ||
        tensor_v->type != GGML_TYPE_F16 || tensor_k->ne[0] != gptoss_key_value_size ||
        tensor_v->ne[0] != gptoss_key_value_size || tensor_k->ne[1] != cache->get_size() ||
        tensor_v->ne[1] != cache->get_size() || tensor_k->ne[2] != cache->get_n_stream() ||
        tensor_v->ne[2] != cache->get_n_stream() ||
        tensor_k->nb[1] != gptoss_key_value_size * sizeof(uint16_t) ||
        tensor_v->nb[1] != gptoss_key_value_size * sizeof(uint16_t) ||
        tensor_k->nb[2] != cache->get_size() * tensor_k->nb[1] ||
        tensor_v->nb[2] != cache->get_size() * tensor_v->nb[1]) {
        return false;
    }

    auto * k_result = static_cast<__half *>(tensor_k->data);
    auto * v_result = static_cast<__half *>(tensor_v->data);
    if (k_result == nullptr || v_result == nullptr) {
        return false;
    }

    k = k_result;
    v = v_result;
    return true;
}
