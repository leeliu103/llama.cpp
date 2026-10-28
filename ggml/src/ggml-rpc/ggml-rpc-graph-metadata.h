#pragma once

#include "ggml-impl.h"

#include <cstdint>
#include <limits>
#include <vector>

// Meta stages retain whole-graph counts, including consumers in later stages.
// A collective adds an external use even when the tensor has no graph consumer.
template<class Visitor>
inline bool rpc_graph_metadata_visit(const ggml_cgraph * graph, const ggml_tensor * allreduce, Visitor visit) {
    if (!graph || graph->n_nodes <= 0 || !graph->use_counts || !graph->visited_hash_set.size) {
        return false;
    }
    bool found_allreduce = allreduce == nullptr;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (!node) return false;
        const size_t slot = ggml_hash_find(&graph->visited_hash_set, node);
        if (slot == GGML_HASHSET_FULL || !ggml_bitset_get(graph->visited_hash_set.used, slot)) return false;
        int32_t uses = graph->use_counts[slot];
        if (uses < 0) return false;
        if (node == allreduce) {
            if (uses == std::numeric_limits<int32_t>::max()) return false;
            ++uses;
            found_allreduce = true;
        }
        visit(node, uses);
    }
    return found_allreduce;
}

inline bool rpc_graph_metadata_collect(const ggml_cgraph * graph, const ggml_tensor * allreduce,
                                       std::vector<int32_t> & counts) {
    counts.clear();
    if (!graph || graph->n_nodes <= 0 || !graph->use_counts || !graph->visited_hash_set.size) return false;
    counts.reserve(graph->n_nodes);
    return rpc_graph_metadata_visit(graph, allreduce, [&](const ggml_tensor *, int32_t uses) {
        counts.push_back(uses);
    });
}

inline void rpc_graph_metadata_hash_value(uint64_t value, uint64_t & hash) {
    for (int byte = 0; byte < 8; ++byte) {
        hash = (hash ^ (value & 255)) * UINT64_C(1099511628211);
        value >>= 8;
    }
}

inline uint64_t rpc_graph_metadata_hash(const uint64_t * ids, const int32_t * counts, size_t count,
                                        uint64_t allreduce_id) {
    uint64_t hash = UINT64_C(14695981039346656037);
    rpc_graph_metadata_hash_value(count, hash);
    rpc_graph_metadata_hash_value(allreduce_id, hash);
    for (size_t i = 0; i < count; ++i) {
        rpc_graph_metadata_hash_value(ids[i], hash);
        rpc_graph_metadata_hash_value(static_cast<uint32_t>(counts[i]), hash);
    }
    return hash;
}

// Stable repeats still read live metadata, without materializing wire arrays.
inline bool rpc_graph_metadata_fingerprint(const ggml_cgraph * graph, const ggml_tensor * allreduce,
                                           uint64_t & fingerprint) {
    fingerprint = 0;
    if (!graph) return false;
    uint64_t hash = UINT64_C(14695981039346656037);
    rpc_graph_metadata_hash_value(graph->n_nodes, hash);
    rpc_graph_metadata_hash_value(reinterpret_cast<uintptr_t>(allreduce), hash);
    if (!rpc_graph_metadata_visit(graph, allreduce, [&](const ggml_tensor * node, int32_t uses) {
            rpc_graph_metadata_hash_value(reinterpret_cast<uintptr_t>(node), hash);
            rpc_graph_metadata_hash_value(static_cast<uint32_t>(uses), hash);
        })) return false;
    fingerprint = hash;
    return true;
}

// The received legacy graph remains untouched. Only this extension view has
// populated use metadata, so a later stock submission keeps its old behavior.
class rpc_graph_metadata {
public:
    rpc_graph_metadata() = default;
    rpc_graph_metadata(const rpc_graph_metadata &) = delete;
    rpc_graph_metadata & operator=(const rpc_graph_metadata &) = delete;
    ~rpc_graph_metadata() { clear(); }

    bool initialize(const ggml_cgraph * source, const int32_t * counts, size_t count,
                    const ggml_tensor * allreduce = nullptr) {
        clear();
        if (!source || source->n_nodes <= 0 || size_t(source->n_nodes) != count || !counts) return false;
        graph_ = *source;
        graph_.visited_hash_set = ggml_hash_set_new(count);
        owned_ = true;
        counts_.assign(graph_.visited_hash_set.size, 0);
        graph_.use_counts = counts_.data();
        std::vector<int32_t> local_uses(graph_.visited_hash_set.size, 0);
        for (size_t i = 0; i < count; ++i) {
            if (!graph_.nodes[i] || counts[i] < 0) return reject();
            const size_t slot = ggml_hash_insert(&graph_.visited_hash_set, graph_.nodes[i]);
            if (slot == GGML_HASHSET_FULL || slot == GGML_HASHSET_ALREADY_EXISTS) return reject();
            counts_[slot] = counts[i];
        }
        if (allreduce) {
            const size_t slot = ggml_hash_find(&graph_.visited_hash_set, allreduce);
            if (slot == GGML_HASHSET_FULL || !ggml_bitset_get(graph_.visited_hash_set.used, slot) ||
                counts_[slot] == 0) return reject();
            local_uses[slot] = 1;
        }
        for (size_t i = 0; i < count; ++i) {
            for (const auto * src : graph_.nodes[i]->src) {
                if (!src) continue;
                const size_t slot = ggml_hash_find(&graph_.visited_hash_set, src);
                if (slot == GGML_HASHSET_FULL || !ggml_bitset_get(graph_.visited_hash_set.used, slot)) continue;
                if (local_uses[slot] == std::numeric_limits<int32_t>::max() ||
                    ++local_uses[slot] > counts_[slot]) return reject();
            }
        }
        // Replacing metadata must invalidate any previously captured executable.
        graph_.uid = ggml_graph_next_uid();
        return true;
    }

    ggml_cgraph * view() { return owned_ ? &graph_ : nullptr; }

private:
    bool reject() { clear(); return false; }
    void clear() {
        if (owned_) ggml_hash_set_free(&graph_.visited_hash_set);
        owned_ = false;
        graph_ = {};
        counts_.clear();
    }
    ggml_cgraph graph_{};
    std::vector<int32_t> counts_;
    bool owned_ = false;
};
