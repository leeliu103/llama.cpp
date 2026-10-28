#pragma once

#include "ggml-backend.h"

struct rpc_execution_extension_hooks {
    ggml_backend_execution_extension_begin_t begin = nullptr;
    ggml_backend_execution_extension_end_t end = nullptr;
    ggml_backend_execution_extension_prepare_v1_t prepare = nullptr;
    ggml_backend_execution_extension_seal_v1_t seal = nullptr;
    ggml_backend_execution_extension_cancel_v1_t cancel = nullptr;
};

// The worker owns synchronization. Destruction only cancels host bookkeeping;
// it must not call the status-reading end procedure after partial submission.
class rpc_execution_extension_scope {
public:
    rpc_execution_extension_scope() = default;
    rpc_execution_extension_scope(const rpc_execution_extension_scope &) = delete;
    rpc_execution_extension_scope & operator=(const rpc_execution_extension_scope &) = delete;
    ~rpc_execution_extension_scope() { cancel(); }

    bool begin(ggml_backend_t backend, const char * plan, const rpc_execution_extension_hooks & hooks,
               bool needs_full_lifecycle) {
        if (backend_ || !backend || !plan || !plan[0] || !hooks.begin || !hooks.end || !hooks.cancel ||
            (needs_full_lifecycle && (!hooks.prepare || !hooks.seal))) return false;
        // A rejected begin does not transfer ownership of an existing scope.
        if (!hooks.begin(backend, plan)) return false;
        backend_ = backend;
        hooks_ = hooks;
        return true;
    }

    bool prepare(size_t n_graphs, ggml_cgraph * const * graphs, ggml_tensor * const * allreduce_operands) {
        if (!backend_ || prepared_ || seal_attempted_ || failed_) return false;
        if (!n_graphs || !graphs || !allreduce_operands) return fail();
        for (size_t i = 0; i < n_graphs; ++i) if (!graphs[i]) return fail();
        if (hooks_.prepare && !hooks_.prepare(backend_, n_graphs, graphs, allreduce_operands)) return fail();
        prepared_ = true;
        return true;
    }

    bool seal() {
        if (!backend_) return true;
        if (!prepared_ || seal_attempted_ || failed_) return fail();
        seal_attempted_ = true;
        sealed_ = !hooks_.seal || hooks_.seal(backend_);
        failed_ = !sealed_;
        return sealed_;
    }

    // Call only after the worker's existing backend synchronization returned.
    bool mark_synchronized() {
        if (!backend_) return true;
        if (!seal_attempted_ || synchronized_) return false;
        synchronized_ = true;
        return true;
    }

    bool finish(uint64_t * custom_stages) {
        if (custom_stages) *custom_stages = 0;
        if (!backend_) return true;
        if (!prepared_ || !sealed_ || !synchronized_ || failed_) {
            cancel();
            return false;
        }
        uint64_t count = 0;
        const bool ok = hooks_.end(backend_, &count);
        reset();
        if (ok && custom_stages) *custom_stages = count;
        return ok;
    }

    template<class Synchronize>
    bool complete(Synchronize synchronize, uint64_t * custom_stages) {
        const bool sealed = seal();
        // A failed final status-copy enqueue must not skip the token's sync.
        synchronize();
        const bool synchronized = mark_synchronized();
        if (!sealed || !synchronized) {
            if (custom_stages) *custom_stages = 0;
            cancel();
            return false;
        }
        return finish(custom_stages);
    }

    void cancel() {
        if (!backend_) return;
        const auto backend = backend_;
        const auto cancel_fn = hooks_.cancel;
        reset();
        cancel_fn(backend);
    }

private:
    bool fail() { failed_ = true; return false; }
    void reset() {
        backend_ = nullptr;
        hooks_ = {};
        prepared_ = seal_attempted_ = sealed_ = synchronized_ = failed_ = false;
    }
    ggml_backend_t backend_ = nullptr;
    rpc_execution_extension_hooks hooks_{};
    bool prepared_ = false;
    bool seal_attempted_ = false;
    bool sealed_ = false;
    bool synchronized_ = false;
    bool failed_ = false;
};
