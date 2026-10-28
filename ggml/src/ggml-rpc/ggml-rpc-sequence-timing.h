#pragma once

// Optional host-elapsed diagnostics only. No protocol or execution decisions.
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace rpc_sequence_timing {

inline bool enabled() {
    static const bool value = [] {
        const char * env = std::getenv("GGML_RPC_SEQUENCE_TIMING");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return value;
}

inline uint64_t now_ns() {
#ifdef GGML_RPC_SEQUENCE_TIMING_TEST_CLOCK
    return GGML_RPC_SEQUENCE_TIMING_TEST_CLOCK();
#else
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

enum phase : size_t {
    total, build, dispatch_join, commit, send, response_wait, receive, setup,
    parse, stock_parse_enqueue, prepare, enqueue, complete, sync, reply, phase_count
};

inline const char * phase_name(size_t value) {
    static const char * const names[] = {
        "total", "build", "dispatch_join", "commit", "send", "response_wait", "receive", "setup",
        "parse", "stock_parse_enqueue", "prepare", "enqueue", "complete", "sync", "reply"
    };
    return names[value];
}

enum outcome : size_t { failed, success, cache_miss };

struct sample {
    uint32_t plan = 0;
    uint64_t sequence_graphs = 0;
    uint64_t bytes = 0;
    uint64_t graphs_seen = 0;
    uint64_t allreduces_seen = 0;
    uint64_t graph_payloads = 0;
    uint64_t metadata_payloads = 0;
    outcome result = failed;
    size_t last_phase = total;
    uint32_t observed = 0;
    std::array<uint64_t, phase_count> ns{};
};

struct moment {
    uint64_t count = 0, sum = 0, min = UINT64_MAX, max = 0;
    void add(uint64_t value) {
        ++count;
        sum += value;
        if (value < min) min = value;
        if (value > max) max = value;
    }
};

class collector {
    struct bucket {
        bool used = false;
        uint32_t plan = 0;
        uint64_t sequence_graphs = 0;
        size_t cache_state = 0;
        outcome result = failed;
        uint64_t requests = 0, bytes = 0, graphs_seen = 0, allreduces_seen = 0;
        uint64_t graph_payloads = 0, metadata_payloads = 0;
        std::array<moment, phase_count> phases{};
        std::array<uint64_t, phase_count> last_phases{};
    };
    std::array<bucket, 64> buckets{};
    uint64_t requests = 0;
    uint64_t overflow = 0;
    uint64_t serial;

public:
    collector() {
        static std::atomic<uint64_t> next{0};
        serial = ++next;
    }

    // One owner thread only; client rank samples are committed after all joins.
    void record(const sample & value) {
        bool plan_seen = false;
        for (const auto & item : buckets) {
            plan_seen = plan_seen || (item.used && item.plan == value.plan);
        }
        const size_t state = value.graph_payloads ? (plan_seen ? 1 : 0) : value.metadata_payloads ? 2 : 3;
        ++requests;
        bucket * target = nullptr;
        bucket * unused = nullptr;
        for (auto & item : buckets) {
            if (!item.used && !unused) unused = &item;
            if (item.used && item.plan == value.plan && item.sequence_graphs == value.sequence_graphs &&
                    item.cache_state == state && item.result == value.result) {
                target = &item;
                break;
            }
        }
        if (!target) {
            target = unused;
            if (!target) { ++overflow; return; }
            target->used = true;
            target->plan = value.plan;
            target->sequence_graphs = value.sequence_graphs;
            target->cache_state = state;
            target->result = value.result;
        }
        ++target->requests;
        target->bytes += value.bytes;
        target->graphs_seen += value.graphs_seen;
        target->allreduces_seen += value.allreduces_seen;
        target->graph_payloads += value.graph_payloads;
        target->metadata_payloads += value.metadata_payloads;
        ++target->last_phases[value.last_phase];
        for (size_t i = 0; i < phase_count; ++i) {
            if (value.observed & (uint32_t(1) << i)) target->phases[i].add(value.ns[i]);
        }
    }

    void emit(const char * owner, int64_t rank, FILE * output = stderr) const {
        if (!requests) return;
        static std::mutex output_mutex;
        const std::lock_guard<std::mutex> lock(output_mutex);
        static const char * const states[] = {"cold_graphs", "changed_graphs", "changed_metadata", "reuse"};
        static const char * const outcomes[] = {"failed", "success", "cache_miss"};
        size_t used_buckets = 0;
        for (const auto & item : buckets) used_buckets += item.used;
        for (const auto & item : buckets) {
            if (!item.used) continue;
            std::fprintf(output, "[RPC sequence timing] {\"version\":1,\"owner\":\"%s\",\"rank\":%" PRId64
                ",\"collector_id\":%" PRIu64 ",\"collector_requests\":%" PRIu64 ",\"collector_overflow_requests\":%" PRIu64
                ",\"collector_buckets\":%zu,\"overflow\":%s"
                ",\"plan\":%u,\"sequence_graphs\":%" PRIu64 ",\"cache_state\":\"%s\",\"outcome\":\"%s\""
                ",\"requests\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"graphs_seen\":%" PRIu64
                ",\"allreduces_seen\":%" PRIu64 ",\"graph_payloads\":%" PRIu64 ",\"metadata_payloads\":%" PRIu64 ",\"phases_ns\":{",
                owner, rank, serial, requests, overflow, used_buckets, overflow ? "true" : "false", item.plan, item.sequence_graphs,
                states[item.cache_state], outcomes[item.result], item.requests, item.bytes, item.graphs_seen,
                item.allreduces_seen, item.graph_payloads, item.metadata_payloads);
            bool first = true;
            for (size_t i = 0; i < phase_count; ++i) {
                const auto & moment = item.phases[i];
                if (!moment.count) continue;
                std::fprintf(output, "%s\"%s\":{\"count\":%" PRIu64 ",\"sum\":%" PRIu64 ",\"min\":%" PRIu64
                    ",\"max\":%" PRIu64 "}", first ? "" : ",", phase_name(i), moment.count, moment.sum, moment.min, moment.max);
                first = false;
            }
            std::fprintf(output, "},\"last_phases\":{");
            first = true;
            for (size_t i = 0; i < phase_count; ++i) {
                if (!item.last_phases[i]) continue;
                std::fprintf(output, "%s\"%s\":%" PRIu64, first ? "" : ",", phase_name(i), item.last_phases[i]);
                first = false;
            }
            std::fprintf(output, "}}\n");
        }
        std::fflush(output);
    }
};

class scope {
    sample & value;
    collector * sink;
    bool active;
    uint64_t begin_ns = 0;
    uint64_t phase_begin_ns = 0;
    size_t current = phase_count;

    void stop_phase(uint64_t now) {
        if (current == phase_count) return;
        value.ns[current] += now - phase_begin_ns;
        value.observed |= uint32_t(1) << current;
    }

public:
    scope(sample & value, collector * sink, bool active) : value(value), sink(sink), active(active) {
        if (active) begin_ns = now_ns();
    }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
    ~scope() {
        if (!active) return;
        const uint64_t now = now_ns();
        stop_phase(now);
        value.ns[total] = now - begin_ns;
        value.observed |= uint32_t(1) << total;
        if (sink) sink->record(value);
    }
    void next(phase which) {
        if (!active) return;
        const uint64_t now = now_ns();
        stop_phase(now);
        current = which;
        value.last_phase = which;
        phase_begin_ns = now;
    }
    template <typename Function>
    void measure(phase which, Function && function) {
        if (!active) { function(); return; }
        const uint64_t before = now_ns();
        function();
        value.ns[which] += now_ns() - before;
        value.observed |= uint32_t(1) << which;
    }
    sample & data() { return value; }
};

} // namespace rpc_sequence_timing
