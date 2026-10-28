#pragma once

struct ggml_cuda_attention_o_region {
    qwen3moe_o_q8_128_args args{};
    ggml_tensor * attention = nullptr;
};

struct ggml_cuda_attention_o_tensor_signature {
    const ggml_tensor * tensor = nullptr;
    const ggml_tensor * view = nullptr;
    const void * data = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    const void * allocation = nullptr;
    size_t allocation_bytes = 0;
    size_t view_offset = 0;
    ggml_type type = GGML_TYPE_F32;
    ggml_op op = GGML_OP_NONE;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    std::array<size_t, GGML_MAX_DIMS> nb{};
    std::array<const ggml_tensor *, GGML_MAX_SRC> sources{};
    std::array<int32_t, GGML_MAX_OP_PARAMS/sizeof(int32_t)> parameters{};

    bool operator==(const ggml_cuda_attention_o_tensor_signature & other) const {
        return tensor == other.tensor && view == other.view && data == other.data && buffer == other.buffer &&
               allocation == other.allocation && allocation_bytes == other.allocation_bytes &&
               view_offset == other.view_offset && type == other.type && op == other.op &&
               ne == other.ne && nb == other.nb && sources == other.sources && parameters == other.parameters;
    }
};

struct ggml_cuda_attention_o_signature {
    std::array<ggml_cuda_attention_o_tensor_signature, 9> tensors{};
    uint32_t * status = nullptr;

    bool operator==(const ggml_cuda_attention_o_signature & other) const {
        return tensors == other.tensors && status == other.status;
    }
};

struct ggml_cuda_attention_o_state {
    std::unordered_map<const ggml_tensor *, ggml_cuda_attention_o_region> regions;
    std::unordered_map<const void *, ggml_cuda_attention_o_signature> signatures;
};

static ggml_cuda_attention_o_tensor_signature ggml_cuda_attention_o_describe(const ggml_tensor * tensor) {
    ggml_cuda_attention_o_tensor_signature result;
    result.tensor = tensor;
    result.view = tensor->view_src;
    result.data = tensor->data;
    result.buffer = tensor->buffer;
    result.allocation = ggml_backend_buffer_get_base(tensor->buffer);
    result.allocation_bytes = ggml_backend_buffer_get_size(tensor->buffer);
    result.view_offset = tensor->view_offs;
    result.type = tensor->type;
    result.op = tensor->op;
    std::copy_n(tensor->ne, GGML_MAX_DIMS, result.ne.begin());
    std::copy_n(tensor->nb, GGML_MAX_DIMS, result.nb.begin());
    std::copy_n(tensor->src, GGML_MAX_SRC, result.sources.begin());
    std::memcpy(result.parameters.data(), tensor->op_params, sizeof(tensor->op_params));
    return result;
}

static bool ggml_cuda_attention_o_match(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int start,
                                       const ggml_tensor * operand,
                                       const qwen3moe_qkv_prefix_q8_128_args & prefix,
                                       ggml_cuda_attention_o_region & region,
                                       ggml_cuda_attention_o_signature & signature) {
    const ggml_op ops[] = {GGML_OP_FLASH_ATTN_EXT, GGML_OP_RESHAPE, GGML_OP_MUL_MAT};
    if (!graph || start < 0 || start > graph->n_nodes - 3 || start + 3 != graph->n_nodes) return false;
    const int output_index = start + 2;
    if (!ggml_can_fuse_subgraph(graph, start, 3, ops, &output_index, 1)) return false;
    auto * attention = graph->nodes[start];
    auto * flat = graph->nodes[start + 1];
    auto * output = graph->nodes[start + 2];
    if (!ggml_cuda_qkv_prefix_sources(attention, 4) || !ggml_cuda_qkv_prefix_sources(flat, 1) ||
        !ggml_cuda_qkv_prefix_sources(output, 2)) return false;
    const auto * weights = output->src[0];
    const int64_t columns = int64_t(prefix.q_heads) * 128;
    if ((columns != 2048 && columns != 4096) ||
        !ggml_cuda_qkv_prefix_shape(attention, GGML_TYPE_F32, 128, prefix.q_heads) || !ggml_is_contiguous(attention) ||
        !ggml_cuda_qkv_prefix_shape(flat, GGML_TYPE_F32, columns) || !ggml_is_contiguous(flat) ||
        !ggml_cuda_qkv_prefix_view(flat, attention, attention) ||
        !ggml_cuda_qkv_prefix_shape(weights, GGML_TYPE_Q4_K, columns, 4096) || !ggml_is_contiguous(weights) ||
        weights->nb[0] != sizeof(block_q4_K) ||
        !ggml_cuda_qkv_prefix_shape(output, GGML_TYPE_F32, 4096) || !ggml_is_contiguous(output) ||
        output->src[1] != flat ||
        !ggml_cuda_qkv_prefix_shape(operand, GGML_TYPE_F32, 4096) || !ggml_is_contiguous(operand) ||
        operand->data != output->data || !prefix.status ||
        attention->src[0]->data != prefix.q_output || attention->src[1]->data != prefix.k_cache ||
        attention->src[2]->data != prefix.v_cache) return false;
    const ggml_tensor * tensors[] = {attention, attention->src[0], attention->src[1], attention->src[2],
                                    attention->src[3], flat, weights, output, operand};
    for (const auto * tensor : tensors) if (!ggml_cuda_qkv_prefix_device(ctx, tensor)) return false;
    for (int source = 0; source < 4; ++source) {
        const auto * input = attention->src[source];
        if (ggml_cuda_qkv_prefix_overlap(attention->data, ggml_nbytes(attention), input->data, ggml_nbytes(input))) return false;
    }
    region = {};
    region.args = {weights->data, static_cast<const float *>(flat->data), static_cast<float *>(output->data),
                   prefix.status, ggml_nbytes(weights), ggml_nbytes(flat), ggml_nbytes(output), weights->nb[1], int(columns)};
    region.attention = attention;
    if (!qwen3moe_o_q8_128_supported(&region.args)) return false;

    // These values survive the local FA/O body: residual input, immutable
    // weights/parameters, masks and owner caches must not be overwritten.
    struct live_span { const void * data; size_t bytes; };
    const live_span live[] = {
        {prefix.input, 4096 * sizeof(float)}, {prefix.input_gamma, 4096 * sizeof(float)},
        {prefix.q_gamma, 128 * sizeof(float)}, {prefix.k_gamma, 128 * sizeof(float)},
        {prefix.wq, prefix.q_weight_bytes}, {prefix.wk, prefix.k_weight_bytes}, {prefix.wv, prefix.v_weight_bytes},
        {prefix.position, sizeof(int32_t)}, {prefix.k_slot, sizeof(int64_t)}, {prefix.v_slot, sizeof(int64_t)},
        {prefix.k_cache, prefix.k_cache_bytes}, {prefix.v_cache, prefix.v_cache_bytes},
        {prefix.normalized, ggml_cuda_qkv_prefix_scratch_bytes}, {prefix.status, sizeof(uint32_t)},
        {attention->src[3]->data, ggml_nbytes(attention->src[3])}, {weights->data, ggml_nbytes(weights)}};
    for (const auto & span : live) {
        if (ggml_cuda_qkv_prefix_overlap(region.args.output, region.args.output_bytes, span.data, span.bytes) ||
            ggml_cuda_qkv_prefix_overlap(region.args.input, region.args.input_bytes, span.data, span.bytes)) return false;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (i >= start && i < start + 3) continue;
        int depth = 0;
        for (const ggml_tensor * alias = graph->nodes[i]->view_src; alias; alias = alias->view_src) {
            if (++depth > 64 || alias == attention || alias == flat) return false;
        }
    }
    signature = {};
    for (size_t i = 0; i < signature.tensors.size(); ++i) signature.tensors[i] = ggml_cuda_attention_o_describe(tensors[i]);
    signature.status = prefix.status;
    return true;
}

static bool ggml_cuda_attention_o_prepare(ggml_backend_cuda_context * ctx, ggml_cuda_attention_o_state & state,
                                         const ggml_cuda_qkv_prefix_state & qkv, size_t count,
                                         ggml_cgraph * const * graphs, ggml_tensor * const * operands) {
    state.regions.clear();
    for (size_t g = 0; g < count; ++g) {
        auto * graph = graphs[g];
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i]->op != GGML_OP_FLASH_ATTN_EXT) continue;
            if (i < 24) return false;
            const auto prefix = qkv.regions.find(graph->nodes[i - 24]);
            if (prefix == qkv.regions.end()) return false;
            ggml_cuda_attention_o_region region;
            ggml_cuda_attention_o_signature signature;
            if (!ggml_cuda_attention_o_match(ctx, graph, i, operands[g], prefix->second, region, signature)) {
                GGML_LOG_ERROR("%s: unsupported attention output at %s\n", __func__, graph->nodes[i]->name);
                return false;
            }
            if (!state.regions.emplace(graph->nodes[i], region).second) return false;
            const void * key = graph->nodes[0];
            const auto old = state.signatures.find(key);
            if (old != state.signatures.end() && !(old->second == signature)) {
#ifdef USE_CUDA_GRAPH
                ctx->cuda_graph_cache().erase(key);
#endif
            }
            state.signatures[key] = signature;
        }
    }
    return state.regions.size() == 94;
}

static int ggml_cuda_attention_o_dispatch(ggml_backend_cuda_context * ctx, const ggml_cuda_attention_o_state & state,
                                         ggml_cgraph * graph, int i, hipStream_t stream) {
    const auto found = state.regions.find(graph->nodes[i]);
    if (found == state.regions.end()) return 0;
    ggml_cuda_flash_attn_ext(*ctx, found->second.attention);
    if (hipGetLastError() != hipSuccess || qwen3moe_o_q8_128_enqueue(&found->second.args, stream) != hipSuccess) return -1;
    ++ctx->execution_extension_launches;
    return 2;
}
