#pragma once

#include <cmath>
#include <cstring>

struct ggml_cuda_qkv_prefix_state {
    void * scratch = nullptr;
    std::unordered_map<const ggml_tensor *, qwen3moe_qkv_prefix_q8_128_args> regions;
    std::unordered_map<const void *, qwen3moe_qkv_prefix_q8_128_args> signatures;
};

static constexpr size_t ggml_cuda_qkv_prefix_scratch_bytes = 40064;

static bool ggml_cuda_qkv_prefix_shape(const ggml_tensor * t, ggml_type type,
                                      int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
    return t && t->type == type && t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == 1;
}

static bool ggml_cuda_qkv_prefix_sources(const ggml_tensor * t, int count) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if ((t->src[i] != nullptr) != (i < count)) return false;
    }
    return true;
}

static bool ggml_cuda_qkv_prefix_view(const ggml_tensor * t, const ggml_tensor * parent,
                                     const ggml_tensor * base) {
    return t->src[0] == parent && t->view_src == base && t->view_offs == 0 && t->data == base->data;
}

static bool ggml_cuda_qkv_prefix_layout(const ggml_tensor * a, const ggml_tensor * b) {
    return a->type == b->type && ggml_are_same_shape(a, b) && std::memcmp(a->nb, b->nb, sizeof(a->nb)) == 0;
}

static bool ggml_cuda_qkv_prefix_device(ggml_backend_cuda_context * ctx, const ggml_tensor * t) {
    if (!t || !t->data || !t->buffer || t->buffer->buft != ggml_backend_cuda_buffer_type(ctx->device)) return false;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer));
    const size_t allocation = ggml_backend_buffer_get_size(t->buffer);
    const uintptr_t data = reinterpret_cast<uintptr_t>(t->data);
    const size_t bytes = ggml_nbytes(t);
    return begin && allocation <= UINTPTR_MAX - begin && data >= begin && data <= begin + allocation &&
           bytes && bytes <= begin + allocation - data;
}

static bool ggml_cuda_qkv_prefix_overlap(const void * p, size_t n, const void * q, size_t m) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(q);
    return !a || !b || n > UINTPTR_MAX - a || m > UINTPTR_MAX - b || (a < b + m && b < a + n);
}

static bool ggml_cuda_qkv_prefix_rope(const ggml_tensor * rms, const ggml_tensor * rope,
                                     qwen3moe_qkv_prefix_rope & out) {
    int32_t p[GGML_MAX_OP_PARAMS/sizeof(int32_t)];
    std::memcpy(p, rope->op_params, sizeof(p));
    if (p[1] != 128 || p[2] != GGML_ROPE_TYPE_NEOX || p[4] <= 0) return false;
    for (int i = 11; i < 15; ++i) if (p[i]) return false;
    float f[6], epsilon;
    std::memcpy(f, p + 5, sizeof(f));
    std::memcpy(&epsilon, rms->op_params, sizeof(epsilon));
    for (float value : f) if (!std::isfinite(value)) return false;
    if (!std::isfinite(epsilon) || epsilon <= 0 || f[0] <= 1 || f[1] <= 0 || f[2] < 0 || f[4] <= 0 || f[5] <= 0) return false;
    out = {epsilon, powf(f[0], -2.0f/p[1]), f[1], f[2], f[3], {0, 0}};
    ggml_rope_yarn_corr_dims(p[1], p[4], f[0], f[4], f[5], out.corr_dims);
    return true;
}

static bool ggml_cuda_qkv_prefix_match(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int start,
                                      const ggml_tensor * operand, void * scratch, uint32_t * error,
                                      qwen3moe_qkv_prefix_q8_128_args & args) {
    const ggml_op ops[27] = {
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM,
        GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
        GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW,
        GGML_OP_SET_ROWS, GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_VIEW, GGML_OP_PERMUTE,
        GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_RESHAPE, GGML_OP_MUL_MAT};
    if (!graph || start < 0 || start > graph->n_nodes - 27 || start + 27 != graph->n_nodes || !scratch || !error) return false;
    ggml_tensor * const * n = graph->nodes + start;
    for (int i = 0; i < 27; ++i) if (!n[i] || n[i]->op != ops[i]) return false;
    for (int i = 0; i < 18; ++i) {
        int depth = 0;
        for (const ggml_tensor * a = n[i]->view_src; a; a = a->view_src) if (++depth > 64) return false;
    }
    const int outputs[] = {start + 6, start + 15, start + 17};
    if (!ggml_can_fuse_subgraph(graph, start, 18, ops, outputs, 3)) return false;
    const int sources[27] = {1,2,2,1,1,2,2,2,1,2,1,1,2,2,1,3,1,3,1,1,1,1,1,1,4,1,2};
    for (int i = 0; i < 27; ++i) if (!ggml_cuda_qkv_prefix_sources(n[i], sources[i])) return false;

    const int64_t heads = n[3]->ne[1], kv = n[10]->ne[1];
    if ((heads != 16 && heads != 32) || kv != heads/16) return false;
    const auto * x = n[0]->src[0];
    const auto * gamma = n[1]->src[1];
    const auto * wq = n[2]->src[0];
    const auto * wv = n[7]->src[0];
    const auto * wk = n[9]->src[0];
    const auto * qgamma = n[5]->src[1];
    const auto * kgamma = n[12]->src[1];
    const auto * position = n[6]->src[1];
    const auto * ki = n[15]->src[1];
    const auto * vi = n[17]->src[1];
    const auto * kc = n[15]->src[2];
    const auto * vc = n[17]->src[2];
    const auto * mask = n[24]->src[3];
    const auto * wo = n[26]->src[0];
    if (!ggml_cuda_qkv_prefix_shape(x, GGML_TYPE_F32, 4096) || !ggml_is_contiguous(x) ||
        !ggml_cuda_qkv_prefix_shape(gamma, GGML_TYPE_F32, 4096) || !ggml_is_contiguous(gamma) ||
        !ggml_cuda_qkv_prefix_shape(qgamma, GGML_TYPE_F32, 128) || !ggml_is_contiguous(qgamma) ||
        !ggml_cuda_qkv_prefix_shape(kgamma, GGML_TYPE_F32, 128) || !ggml_is_contiguous(kgamma) ||
        !ggml_cuda_qkv_prefix_shape(wq, GGML_TYPE_Q4_K, 4096, heads*128) || !ggml_is_contiguous(wq) ||
        !ggml_cuda_qkv_prefix_shape(wk, GGML_TYPE_Q4_K, 4096, kv*128) || !ggml_is_contiguous(wk) ||
        (wv->type != GGML_TYPE_Q4_K && wv->type != GGML_TYPE_Q6_K) ||
        !ggml_cuda_qkv_prefix_shape(wv, wv->type, 4096, kv*128) || !ggml_is_contiguous(wv) ||
        !ggml_cuda_qkv_prefix_shape(position, GGML_TYPE_I32, 1) || !ggml_is_contiguous(position) ||
        !ggml_cuda_qkv_prefix_shape(ki, GGML_TYPE_I64, 1) || !ggml_is_contiguous(ki) ||
        !ggml_cuda_qkv_prefix_shape(vi, GGML_TYPE_I64, 1) || !ggml_is_contiguous(vi)) return false;
    for (int i : {0,1}) if (!ggml_cuda_qkv_prefix_shape(n[i], GGML_TYPE_F32, 4096) || !ggml_is_contiguous(n[i])) return false;
    for (int i : {2}) if (!ggml_cuda_qkv_prefix_shape(n[i], GGML_TYPE_F32, heads*128) || !ggml_is_contiguous(n[i])) return false;
    for (int i : {3,4,5,6}) if (!ggml_cuda_qkv_prefix_shape(n[i], GGML_TYPE_F32, 128, heads) || !ggml_is_contiguous(n[i])) return false;
    for (int i : {7,9,14,16}) if (!ggml_cuda_qkv_prefix_shape(n[i], GGML_TYPE_F32, kv*128) || !ggml_is_contiguous(n[i])) return false;
    for (int i : {8,10,11,12,13}) if (!ggml_cuda_qkv_prefix_shape(n[i], GGML_TYPE_F32, 128, kv) || !ggml_is_contiguous(n[i])) return false;
    if (n[1]->src[0] != n[0] || n[2]->src[1] != n[1] || n[4]->src[0] != n[3] || n[5]->src[0] != n[4] ||
        n[6]->src[0] != n[5] || n[7]->src[1] != n[1] || n[9]->src[1] != n[1] || n[11]->src[0] != n[10] ||
        n[12]->src[0] != n[11] || n[13]->src[0] != n[12] || n[13]->src[1] != position ||
        n[15]->src[0] != n[14] || n[17]->src[0] != n[16] ||
        !ggml_cuda_qkv_prefix_view(n[3], n[2], n[2]) || !ggml_cuda_qkv_prefix_view(n[8], n[7], n[7]) ||
        !ggml_cuda_qkv_prefix_view(n[10], n[9], n[9]) || !ggml_cuda_qkv_prefix_view(n[14], n[13], n[13]) ||
        !ggml_cuda_qkv_prefix_view(n[16], n[8], n[7])) return false;
    const int64_t capacity = kc->ne[1], visible = n[20]->ne[2];
    if (capacity <= 0 || capacity > 32768 || visible <= 0 || visible > capacity ||
        !ggml_cuda_qkv_prefix_shape(kc, GGML_TYPE_F16, kv*128, capacity) || !ggml_is_contiguous(kc) ||
        !ggml_cuda_qkv_prefix_shape(vc, GGML_TYPE_F16, kv*128, capacity) || !ggml_is_contiguous(vc) ||
        !ggml_cuda_qkv_prefix_view(n[15], n[14], kc) || !ggml_cuda_qkv_prefix_view(n[17], n[16], vc) ||
        !ggml_cuda_qkv_prefix_layout(n[15], kc) || !ggml_cuda_qkv_prefix_layout(n[17], vc)) return false;
    if (!ggml_cuda_qkv_prefix_view(n[18], n[6], n[6]) || !ggml_cuda_qkv_prefix_view(n[19], n[18], n[6]) ||
        !ggml_cuda_qkv_prefix_shape(n[18], GGML_TYPE_F32, 128, heads) || !ggml_is_contiguous(n[18]) ||
        !ggml_cuda_qkv_prefix_shape(n[19], GGML_TYPE_F32, 128, 1, heads) ||
        n[19]->nb[0] != 4 || n[19]->nb[1] != size_t(heads*512) || n[19]->nb[2] != 512 || n[19]->nb[3] != size_t(heads*512)) return false;
    for (int i : {20,22}) {
        const auto * cache = i == 20 ? kc : vc;
        if (!ggml_cuda_qkv_prefix_view(n[i], cache, cache) || !ggml_cuda_qkv_prefix_view(n[i+1], n[i], cache) ||
            !ggml_cuda_qkv_prefix_shape(n[i], GGML_TYPE_F16, 128, kv, visible) ||
            !ggml_cuda_qkv_prefix_shape(n[i+1], GGML_TYPE_F16, 128, visible, kv) ||
            n[i]->nb[0] != 2 || n[i]->nb[1] != 256 || n[i]->nb[2] != cache->nb[1] || n[i]->nb[3] != cache->nb[2] ||
            n[i+1]->nb[0] != 2 || n[i+1]->nb[1] != cache->nb[1] || n[i+1]->nb[2] != 256 || n[i+1]->nb[3] != cache->nb[2]) return false;
    }
    if (n[24]->src[0] != n[19] || n[24]->src[1] != n[21] || n[24]->src[2] != n[23] ||
        !ggml_cuda_qkv_prefix_shape(mask, GGML_TYPE_F16, visible) || !ggml_is_contiguous(mask) ||
        !ggml_cuda_qkv_prefix_shape(n[24], GGML_TYPE_F32, 128, heads) || !ggml_is_contiguous(n[24]) ||
        !ggml_cuda_qkv_prefix_view(n[25], n[24], n[24]) || !ggml_cuda_qkv_prefix_shape(n[25], GGML_TYPE_F32, heads*128) ||
        n[26]->src[1] != n[25] || !ggml_cuda_qkv_prefix_shape(n[26], GGML_TYPE_F32, 4096) || !ggml_is_contiguous(n[26]) ||
        wo->ne[0] != heads*128 || wo->ne[1] != 4096 || wo->ne[2] != 1 || wo->ne[3] != 1 ||
        !ggml_cuda_qkv_prefix_shape(operand, GGML_TYPE_F32, 4096) || !ggml_is_contiguous(operand) || operand->data != n[26]->data ||
        !ggml_cuda_qkv_prefix_device(ctx, operand)) return false;

    // Source-use metadata is whole-graph metadata. Also reject escaping storage
    // aliases that are not represented by src[] use counts.
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (i >= start && i < start + 18) continue;
        int depth = 0;
        for (const ggml_tensor * a = graph->nodes[i]->view_src; a; a = a->view_src) {
            if (++depth > 64) return false;
            for (int j = 0; j < 18; ++j) if (a == n[j] && j != 6 && j != 15 && j != 17) return false;
        }
    }
    for (const auto * t : {x,gamma,wq,wk,wv,qgamma,kgamma,position,ki,vi,mask,wo}) {
        if (!ggml_cuda_qkv_prefix_device(ctx, t)) return false;
        int depth = 0;
        for (const ggml_tensor * a = t; a; a = a->view_src) {
            if (++depth > 64) return false;
            for (int j = 0; j < 18; ++j) if (a == n[j]) return false;
        }
    }
    for (const auto * t : {static_cast<const ggml_tensor *>(n[6]),kc,vc}) if (!ggml_cuda_qkv_prefix_device(ctx, t)) return false;
    args = {};
    args.wq = wq->data; args.wk = wk->data; args.wv = wv->data;
    args.q_stride = wq->nb[1]; args.k_stride = wk->nb[1]; args.v_stride = wv->nb[1];
    args.q_weight_bytes = ggml_nbytes(wq); args.k_weight_bytes = ggml_nbytes(wk); args.v_weight_bytes = ggml_nbytes(wv);
    args.input = static_cast<const float *>(x->data); args.input_gamma = static_cast<const float *>(gamma->data);
    args.q_gamma = static_cast<const float *>(qgamma->data); args.k_gamma = static_cast<const float *>(kgamma->data);
    args.position = static_cast<const int32_t *>(position->data);
    args.k_slot = static_cast<const int64_t *>(ki->data); args.v_slot = static_cast<const int64_t *>(vi->data);
    args.q_heads = int(heads); args.kv_heads = int(kv); args.v_q6 = wv->type == GGML_TYPE_Q6_K;
    std::memcpy(&args.input_epsilon, n[0]->op_params, sizeof(float));
    if (!ggml_cuda_qkv_prefix_rope(n[4], n[6], args.q_rope) || !ggml_cuda_qkv_prefix_rope(n[11], n[13], args.k_rope)) return false;
    auto * p = static_cast<uint8_t *>(scratch);
    args.normalized = reinterpret_cast<float *>(p); p += 16384;
    args.q_projected = reinterpret_cast<float *>(p); p += 16384;
    args.k_projected = reinterpret_cast<float *>(p); p += 1024;
    args.v_projected = reinterpret_cast<float *>(p); p += 1024;
    args.k_rotated = reinterpret_cast<float *>(p); p += 1024;
    args.xq = reinterpret_cast<int8_t *>(p); p += 4096;
    args.xd = reinterpret_cast<float *>(p);
    args.q_output = static_cast<float *>(n[6]->data);
    args.k_cache = static_cast<ggml_fp16_t *>(kc->data); args.v_cache = static_cast<ggml_fp16_t *>(vc->data);
    args.k_row_bytes = kc->nb[1]; args.v_row_bytes = vc->nb[1];
    args.k_cache_bytes = ggml_nbytes(kc); args.v_cache_bytes = ggml_nbytes(vc);
    args.cache_rows = capacity; args.status = error;
    if (!qwen3moe_qkv_prefix_q8_128_supported(&args)) return false;
    for (const auto * live : {mask,wo}) {
        for (const auto * write : {static_cast<const ggml_tensor *>(n[6]),kc,vc}) if (ggml_cuda_qkv_prefix_overlap(write->data, ggml_nbytes(write), live->data, ggml_nbytes(live))) return false;
        if (ggml_cuda_qkv_prefix_overlap(scratch, ggml_cuda_qkv_prefix_scratch_bytes, live->data, ggml_nbytes(live)) ||
            ggml_cuda_qkv_prefix_overlap(error, sizeof(uint32_t), live->data, ggml_nbytes(live))) return false;
    }
    return true;
}

static bool ggml_cuda_qkv_prefix_prepare(ggml_backend_cuda_context * ctx, ggml_cuda_qkv_prefix_state & state,
                                        size_t count, ggml_cgraph * const * graphs, ggml_tensor * const * operands,
                                        uint32_t * error) {
    if (!state.scratch && hipMalloc(&state.scratch, ggml_cuda_qkv_prefix_scratch_bytes) != hipSuccess) return false;
    state.regions.clear();
    for (size_t g = 0; g < count; ++g) {
        auto * graph = graphs[g];
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i]->op != GGML_OP_FLASH_ATTN_EXT) continue;
            qwen3moe_qkv_prefix_q8_128_args args{};
            const int start = i - 24;
            if (!ggml_cuda_qkv_prefix_match(ctx, graph, start, operands[g], state.scratch, error, args)) {
                GGML_LOG_ERROR("%s: unsupported attention prefix at %s\n", __func__, graph->nodes[i]->name);
                return false;
            }
            if (!state.regions.emplace(graph->nodes[start], args).second) return false;
            const void * key = graph->nodes[0];
            const auto old = state.signatures.find(key);
            if (old != state.signatures.end() && std::memcmp(&old->second, &args, sizeof(args)) != 0) {
#ifdef USE_CUDA_GRAPH
                ctx->cuda_graph_cache().erase(key);
#endif
            }
            state.signatures[key] = args;
        }
    }
    // Q/K/V cache updates remain present in all94 layers even without logits.
    return state.regions.size() == 94;
}

static int ggml_cuda_qkv_prefix_dispatch(ggml_backend_cuda_context * ctx, const ggml_cuda_qkv_prefix_state & state,
                                        ggml_cgraph * graph, int i, hipStream_t stream) {
    const auto found = state.regions.find(graph->nodes[i]);
    if (found == state.regions.end()) return 0;
    if (qwen3moe_qkv_prefix_q8_128_enqueue(&found->second, stream) != hipSuccess) return -1;
    ++ctx->execution_extension_launches;
    return 17;
}
