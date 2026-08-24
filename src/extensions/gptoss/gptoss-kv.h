#pragma once

#include <hip/hip_fp16.h>

#include <cstdint>
#include <vector>

class llama_kv_cache;
class llama_kv_cache_iswa_context;
struct llama_ubatch;

struct gptoss_kv_cache_rows {
    const llama_kv_cache * cache = nullptr;

    std::vector<int64_t> write_rows;
    std::vector<int32_t> read_rows;
    std::vector<int32_t> sequence_lengths;
};

struct gptoss_kv_batch {
    gptoss_kv_cache_rows base;
    gptoss_kv_cache_rows swa;

    std::vector<int32_t> query_offsets;
};

struct gptoss_fa_block_table {
    std::vector<int32_t> row_indices;
    int64_t              stride = 0;
};

bool gptoss_build_kv_batch(const llama_ubatch &                  ubatch,
                           const llama_kv_cache_iswa_context &   context,
                           gptoss_kv_batch &                     batch);

bool gptoss_build_fa_block_table(const gptoss_kv_cache_rows & rows, gptoss_fa_block_table & table);

bool gptoss_get_kv_storage(const llama_kv_cache * cache, uint32_t layer, __half *& k, __half *& v);
