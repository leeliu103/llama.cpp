#pragma once

// Included after ggml_backend_cuda_context is complete.
struct ggml_cuda_full_moe_region {
    qwen3moe_full_moe_args args{};
    qwen3moe_full_moe_launch_info info{};
    qwen3moe_full_moe_q8_256_args native_args{};
    qwen3moe_full_moe_q8_256_launch_info native_info{};
    qwen3moe_full_moe_q8_128_args native128_args{};
    qwen3moe_full_moe_q8_128_launch_info native128_info{};
    qwen3moe_split_moe_q8_128_args split128_args{};
    qwen3moe_split_moe_q8_128_launch_info split128_info{};
    qwen3moe_ffn_router_q8_128_args ffn128_args{};
    qwen3moe_ffn_router_q8_128_launch_info ffn128_info{};
    bool native = false;
    bool native128 = false;
    bool split128 = false;
    bool ffn128 = false;
    bool ffn_direct = false;
    ggml_tensor * output = nullptr;
};

struct ggml_cuda_full_moe_state {
    ggml_cuda_qkv_prefix_state qkv;
    ggml_cuda_qkv_prefix_state qkv_o;
    ggml_cuda_attention_o_state attention_o;
    void * device_storage = nullptr;
    void * ffn_prefix_storage = nullptr;
    uint32_t * host_status = nullptr;
    hipStream_t stream = nullptr;
    bool prepared = false;
    bool submitted = false;
    bool sealed = false;
    bool poisoned = false;
    std::array<bool, 2> queried{};
    std::array<qwen3moe_full_moe_launch_info, 2> query_info{};
    std::array<bool, 2> native_queried{};
    std::array<qwen3moe_full_moe_q8_256_launch_info, 2> native_query_info{};
    std::array<bool, 2> native128_queried{};
    std::array<qwen3moe_full_moe_q8_128_launch_info, 2> native128_query_info{};
    std::array<bool, 2> split128_queried{};
    std::array<qwen3moe_split_moe_q8_128_launch_info, 2> split128_query_info{};
    std::array<bool, 2> ffn128_queried{};
    std::array<qwen3moe_ffn_router_q8_128_launch_info, 2> ffn128_query_info{};
    std::unordered_map<const ggml_tensor *, ggml_cuda_full_moe_region> regions;

    float * activation() const { return static_cast<float *>(device_storage); }
    float * output() const { return activation() + 4096; }
    uint32_t * arrival() const { return reinterpret_cast<uint32_t *>(output() + 4096); }
    uint32_t * error() const { return arrival() + 1; }
    float * router_logits() const { return static_cast<float *>(ffn_prefix_storage); }
    int32_t * selected_experts() const { return reinterpret_cast<int32_t *>(router_logits() + 128); }
    float * routing_weights() const { return reinterpret_cast<float *>(selected_experts() + 8); }
    uint32_t * router_arrival() const { return reinterpret_cast<uint32_t *>(routing_weights() + 8); }
};

static bool ggml_cuda_full_moe_status(hipError_t error, const char * operation) {
    if (error == hipSuccess) {
        return true;
    }
    GGML_LOG_ERROR("%s: %s failed: %s\n", __func__, operation, hipGetErrorString(error));
    return false;
}

static bool ggml_cuda_full_moe_shape(const ggml_tensor * t, ggml_type type,
                                    int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
    return t && t->type == type && t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == 1;
}

struct ggml_cuda_full_moe_binding {
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up = nullptr;
    const ggml_tensor * down = nullptr;
    const ggml_tensor * input = nullptr;
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * routing = nullptr;
    ggml_tensor * output = nullptr;
};

static bool ggml_cuda_full_moe_structure(ggml_cgraph * graph, int i, ggml_cuda_full_moe_binding & binding) {
    const ggml_op ops[] = {GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
        GGML_OP_MUL_MAT_ID, GGML_OP_MUL,
        GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
        GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD};
    if (!graph || i < 0 || i > graph->n_nodes - 20) {
        return false;
    }
    const int out_node = i + 19;
    if (!ggml_can_fuse_subgraph(graph, i, 20, ops, &out_node, 1)) {
        return false;
    }
    const ggml_tensor * glu = graph->nodes[i + 2];
    const ggml_tensor * gate_op = glu->src[0];
    const ggml_tensor * up_op = glu->src[1];
    if (!gate_op || !up_op || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
        !((gate_op == graph->nodes[i] && up_op == graph->nodes[i + 1]) ||
          (gate_op == graph->nodes[i + 1] && up_op == graph->nodes[i]))) {
        return false;
    }
    const ggml_tensor * gate = gate_op->src[0];
    const ggml_tensor * up = up_op->src[0];
    const ggml_tensor * input = gate_op->src[1];
    const ggml_tensor * ids = gate_op->src[2];
    const ggml_tensor * down_op = graph->nodes[i + 3];
    const ggml_tensor * down = down_op->src[0];
    const ggml_tensor * weighted = graph->nodes[i + 4];
    const ggml_tensor * routing = weighted->src[1];
    if (!ggml_cuda_full_moe_shape(gate, GGML_TYPE_Q4_K, 4096, 512, 128) ||
        !ggml_cuda_full_moe_shape(up, GGML_TYPE_Q4_K, 4096, 512, 128) ||
        !ggml_cuda_full_moe_shape(input, GGML_TYPE_F32, 4096) ||
        !ggml_cuda_full_moe_shape(ids, GGML_TYPE_I32, 8) ||
        !ggml_cuda_full_moe_shape(gate_op, GGML_TYPE_F32, 512, 8) ||
        !ggml_cuda_full_moe_shape(up_op, GGML_TYPE_F32, 512, 8) ||
        !ggml_cuda_full_moe_shape(glu, GGML_TYPE_F32, 512, 8) ||
        !down || (down->type != GGML_TYPE_Q4_K && down->type != GGML_TYPE_Q6_K) ||
        !ggml_cuda_full_moe_shape(down, down->type, 512, 4096, 128) ||
        !ggml_cuda_full_moe_shape(down_op, GGML_TYPE_F32, 4096, 8) ||
        !ggml_cuda_full_moe_shape(weighted, GGML_TYPE_F32, 4096, 8) ||
        !ggml_cuda_full_moe_shape(routing, GGML_TYPE_F32, 1, 8) ||
        up_op->src[1] != input || up_op->src[2] != ids || down_op->src[1] != glu || down_op->src[2] != ids ||
        weighted->src[0] != down_op || input->nb[0] != 4 || ids->nb[0] != 4 || routing->nb[0] != 4 ||
        gate->nb[0] != sizeof(block_q4_K) || up->nb[0] != sizeof(block_q4_K) ||
        down->nb[0] != ggml_type_size(down->type) || weighted->nb[0] != 4) {
        return false;
    }
    for (const ggml_tensor * tensor : {gate_op, up_op, glu, down_op, weighted}) {
        if (!ggml_is_contiguous(tensor)) {
            return false;
        }
    }
    const ggml_tensor * previous = graph->nodes[i + 5];
    for (int e = 0; e < 8; ++e) {
        const ggml_tensor * view = graph->nodes[i + 5 + e];
        if (!ggml_cuda_full_moe_shape(view, GGML_TYPE_F32, 4096) || view->src[0] != weighted ||
            view->view_src != weighted || view->view_offs != size_t(e) * weighted->nb[1] ||
            view->nb[0] != 4 || view->nb[1] != weighted->nb[2]) {
            return false;
        }
        if (e > 0) {
            const ggml_tensor * add = graph->nodes[i + 12 + e];
            if (!ggml_cuda_full_moe_shape(add, GGML_TYPE_F32, 4096) ||
                add->src[0] != previous || add->src[1] != view || add->nb[0] != 4) {
                return false;
            }
            previous = add;
        }
    }
    if (!ggml_is_contiguous(graph->nodes[out_node])) {
        return false;
    }
    binding = {gate, up, down, input, ids, routing, graph->nodes[out_node]};
    return true;
}

static bool ggml_cuda_full_moe_match(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int i,
                                    ggml_cuda_full_moe_region & region) {
    ggml_cuda_full_moe_binding binding;
    if (!ggml_cuda_full_moe_structure(graph, i, binding)) {
        return false;
    }
    const auto * gate = binding.gate;
    const auto * up = binding.up;
    const auto * down = binding.down;
    const auto * input = binding.input;
    const auto * ids = binding.ids;
    const auto * routing = binding.routing;
    region.output = binding.output;
    const auto buft = ggml_backend_cuda_buffer_type(ctx->device);
    for (const ggml_tensor * tensor : {gate, up, down, input, ids, routing, static_cast<const ggml_tensor *>(region.output)}) {
        if (!tensor->buffer || tensor->buffer->buft != buft || !tensor->data) {
            return false;
        }
    }
    auto & state = *ctx->full_moe;
    region.args = {gate->data, up->data, down->data, static_cast<const float *>(input->data),
        static_cast<const int32_t *>(ids->data), static_cast<const float *>(routing->data),
        state.output(), state.activation(), state.arrival(), state.error(), down->type,
        gate->nb[1], gate->nb[2], up->nb[1], up->nb[2], down->nb[1], down->nb[2], routing->nb[1], 128, 2};
    region.native = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_full_moe_q8_256;
    region.native128 = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_full_moe_q8_128;
    region.split128 = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_q8_128 ||
                      ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_q8_128 ||
                      ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_o_q8_128;
    if (region.split128) {
        region.split128_args = {gate->data, up->data, down->data, static_cast<const float *>(input->data),
            static_cast<const int32_t *>(ids->data), static_cast<const float *>(routing->data),
            state.output(), state.activation(), state.arrival(), state.error(), down->type,
            gate->nb[1], gate->nb[2], up->nb[1], up->nb[2], down->nb[1], down->nb[2], routing->nb[1], 32, 4};
        return qwen3moe_split_moe_q8_128_supported(&region.split128_args) != 0;
    }
    if (region.native128) {
        region.native128_args = {gate->data, up->data, down->data, static_cast<const float *>(input->data),
            static_cast<const int32_t *>(ids->data), static_cast<const float *>(routing->data),
            state.output(), state.activation(), state.arrival(), state.error(), down->type,
            gate->nb[1], gate->nb[2], up->nb[1], up->nb[2], down->nb[1], down->nb[2], routing->nb[1], 32, 4};
        return qwen3moe_full_moe_q8_128_supported(&region.native128_args) != 0;
    }
    if (region.native) {
        region.native_args = {gate->data, up->data, down->data, static_cast<const float *>(input->data),
            static_cast<const int32_t *>(ids->data), static_cast<const float *>(routing->data),
            state.output(), state.activation(), state.arrival(), state.error(), down->type,
            gate->nb[1], gate->nb[2], up->nb[1], up->nb[2], down->nb[1], down->nb[2], routing->nb[1], 32, 4};
        return qwen3moe_full_moe_q8_256_supported(&region.native_args) != 0;
    }
    return qwen3moe_full_moe_supported(&region.args) != 0;
}

struct ggml_cuda_full_ffn_binding {
    ggml_cuda_full_moe_binding suffix;
    const ggml_tensor * input = nullptr;
    const ggml_tensor * gamma = nullptr;
    const ggml_tensor * router = nullptr;
    float epsilon = 0.0f;
};

static bool ggml_cuda_full_ffn_sources(const ggml_tensor * node, int count) {
    for (int j = 0; j < count; ++j) {
        if (!node->src[j]) {
            return false;
        }
    }
    for (int j = count; j < GGML_MAX_SRC; ++j) {
        if (node->src[j]) {
            return false;
        }
    }
    return true;
}

static bool ggml_cuda_full_ffn_view(const ggml_tensor * view, const ggml_tensor * parent) {
    return view->src[0] == parent && view->view_src == parent && view->view_offs == 0;
}

static bool ggml_cuda_full_ffn_structure(ggml_cgraph * graph, int start, ggml_cuda_full_ffn_binding & binding) {
    if (!graph || start < 0 || start > graph->n_nodes - 34) {
        return false;
    }
    const ggml_op ops[] = {
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,
        GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SUM_ROWS,
        GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_RESHAPE,
        GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU, GGML_OP_MUL_MAT_ID, GGML_OP_MUL,
        GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
        GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
        GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
    };
    int indices[34];
    for (int j = 0; j < 34; ++j) {
        indices[j] = start + j;
    }
    const int output = start + 33;
    // The contiguous helper has a 31-node limit.
    if (!ggml_can_fuse_subgraph_ext(graph, indices, 34, ops, &output, 1)) {
        return false;
    }
    ggml_cuda_full_ffn_binding candidate;
    if (!ggml_cuda_full_moe_structure(graph, start + 14, candidate.suffix)) {
        return false;
    }
    ggml_tensor * const * n = graph->nodes + start;
    const int source_counts[34] = {1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1,
                                   3, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
    for (int j = 0; j < 34; ++j) {
        if (!ggml_cuda_full_ffn_sources(n[j], source_counts[j])) {
            return false;
        }
    }
    candidate.input = n[0]->src[0];
    candidate.gamma = n[1]->src[1];
    candidate.router = n[2]->src[0];
    candidate.epsilon = ggml_get_op_params_f32(n[0], 0);
    for (const ggml_tensor * input : {candidate.input, candidate.gamma, candidate.router,
                                     candidate.suffix.gate, candidate.suffix.up, candidate.suffix.down}) {
        int depth = 0;
        for (const ggml_tensor * ancestor = input; ancestor; ancestor = ancestor->view_src) {
            if (++depth > 64) {
                return false;
            }
            for (int j = 0; j < 34; ++j) {
                if (ancestor == n[j]) {
                    return false;
                }
            }
        }
    }
    if (!ggml_cuda_full_moe_shape(candidate.input, GGML_TYPE_F32, 4096) ||
        !ggml_cuda_full_moe_shape(candidate.gamma, GGML_TYPE_F32, 4096) ||
        !ggml_cuda_full_moe_shape(candidate.router, GGML_TYPE_F32, 4096, 128) ||
        !ggml_is_contiguous(candidate.input) || !ggml_is_contiguous(candidate.gamma) ||
        candidate.router->nb[0] != sizeof(float) ||
        !std::isfinite(candidate.epsilon) || candidate.epsilon <= 0.0f ||
        !ggml_cuda_full_moe_shape(n[0], GGML_TYPE_F32, 4096) ||
        !ggml_cuda_full_moe_shape(n[1], GGML_TYPE_F32, 4096) ||
        !ggml_cuda_full_moe_shape(n[2], GGML_TYPE_F32, 128) ||
        !ggml_cuda_full_moe_shape(n[3], GGML_TYPE_F32, 128) ||
        !ggml_cuda_full_moe_shape(n[4], GGML_TYPE_F32, 1, 128) ||
        !ggml_cuda_full_moe_shape(n[5], GGML_TYPE_I32, 128) ||
        !ggml_cuda_full_moe_shape(n[6], GGML_TYPE_I32, 8) ||
        !ggml_cuda_full_moe_shape(n[7], GGML_TYPE_F32, 1, 8) ||
        !ggml_cuda_full_moe_shape(n[8], GGML_TYPE_F32, 8) ||
        !ggml_cuda_full_moe_shape(n[9], GGML_TYPE_F32, 1) ||
        !ggml_cuda_full_moe_shape(n[10], GGML_TYPE_F32, 1) ||
        !ggml_cuda_full_moe_shape(n[11], GGML_TYPE_F32, 8) ||
        !ggml_cuda_full_moe_shape(n[12], GGML_TYPE_F32, 1, 8) ||
        !ggml_cuda_full_moe_shape(n[13], GGML_TYPE_F32, 4096) ||
        n[1]->src[0] != n[0] || n[2]->src[1] != n[1] || n[3]->src[0] != n[2] ||
        !ggml_cuda_full_ffn_view(n[4], n[3]) || n[5]->src[0] != n[3] ||
        !ggml_cuda_full_ffn_view(n[6], n[5]) || n[6]->nb[0] != sizeof(int32_t) ||
        n[7]->src[0] != n[4] || n[7]->src[1] != n[6] ||
        !ggml_cuda_full_ffn_view(n[8], n[7]) || n[9]->src[0] != n[8] ||
        !ggml_cuda_full_ffn_view(n[10], n[9]) || n[11]->src[0] != n[8] || n[11]->src[1] != n[10] ||
        !ggml_cuda_full_ffn_view(n[12], n[11]) || !ggml_cuda_full_ffn_view(n[13], n[1]) ||
        candidate.suffix.input != n[13] || candidate.suffix.ids != n[6] || candidate.suffix.routing != n[12] ||
        ggml_get_op_params_f32(n[3], 0) != 1.0f || ggml_get_op_params_f32(n[3], 1) != 0.0f ||
        ggml_get_op_params_i32(n[5], 0) != GGML_SORT_ORDER_DESC ||
        ggml_get_op_params_f32(n[10], 0) != 6.103515625e-5f ||
        ggml_get_op_params_f32(n[10], 1) != INFINITY) {
        return false;
    }
    for (int j = 0; j < 14; ++j) {
        if (!ggml_is_contiguous(n[j])) {
            return false;
        }
    }
    binding = candidate;
    return true;
}

static bool ggml_cuda_full_moe_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
    const size_t nx = ggml_nbytes(a), ny = ggml_nbytes(b);
    return x <= UINTPTR_MAX - nx && y <= UINTPTR_MAX - ny && (x + nx <= y || y + ny <= x);
}

static bool ggml_cuda_full_ffn_storage_external(ggml_cgraph * graph, int start,
                                               const ggml_cuda_full_ffn_binding & binding) {
    if (!graph || start < 0 || start > graph->n_nodes - 34) {
        return false;
    }
    const auto & suffix = binding.suffix;
    if (!suffix.output || !suffix.output->data) {
        return false;
    }
    for (const ggml_tensor * tensor : {suffix.gate, suffix.up, suffix.down, binding.input, binding.gamma, binding.router}) {
        if (!tensor || !tensor->data || !ggml_cuda_full_moe_disjoint(suffix.output, tensor)) {
            return false;
        }
        for (int j = 0; j < 33; ++j) {
            const auto * interior = graph->nodes[start + j];
            if (interior->data && !ggml_cuda_full_moe_disjoint(interior, tensor)) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_cuda_full_ffn_match(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int start,
                                    ggml_cuda_full_moe_region & region) {
    ggml_cuda_full_ffn_binding binding;
    if (!ggml_cuda_full_ffn_structure(graph, start, binding)) {
        return false;
    }
    const auto & suffix = binding.suffix;
    const auto buft = ggml_backend_cuda_buffer_type(ctx->device);
    for (const ggml_tensor * tensor : {suffix.gate, suffix.up, suffix.down, binding.input, binding.gamma,
                                      binding.router, static_cast<const ggml_tensor *>(suffix.output)}) {
        if (!tensor->buffer || tensor->buffer->buft != buft || !tensor->data) {
            return false;
        }
    }
    auto & state = *ctx->full_moe;
    region.ffn_direct = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_ffn_router_q8_128_direct;
    // Validate the actual graph destination against every kernel read/scratch
    // span before admitting the direct-output plan. The other plans keep their
    // dedicated staging output and existing copy semantics.
    float * output = region.ffn_direct ? static_cast<float *>(suffix.output->data) : state.output();
    region.ffn128_args = {suffix.gate->data, suffix.up->data, suffix.down->data,
        static_cast<const float *>(binding.input->data), static_cast<const float *>(binding.gamma->data),
        static_cast<const float *>(binding.router->data), output, state.activation(), state.router_logits(),
        state.selected_experts(), state.routing_weights(), state.arrival(), state.router_arrival(), state.error(),
        suffix.down->type, suffix.gate->nb[1], suffix.gate->nb[2], suffix.up->nb[1], suffix.up->nb[2],
        suffix.down->nb[1], suffix.down->nb[2], binding.router->nb[1], binding.epsilon, 32, 4};
    if (!qwen3moe_ffn_router_q8_128_supported(&region.ffn128_args)) {
        return false;
    }
    if (!ggml_cuda_full_ffn_storage_external(graph, start, binding)) {
        return false;
    }
    region.ffn128 = true;
    region.output = suffix.output;
    return true;
}

static bool ggml_cuda_full_moe_prepare(ggml_backend_cuda_context * ctx, size_t count,
                                      ggml_cgraph * const * graphs, ggml_tensor * const * operands) {
    auto & state = *ctx->full_moe;
    if (state.poisoned || state.prepared || state.submitted || ctx->curr_stream_no != 0 || !count || !graphs || !operands) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    state.stream = ctx->stream();
    const bool full_ffn = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_ffn_router_q8_128 ||
                          ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_ffn_router_q8_128_direct;
    if (!state.device_storage && !ggml_cuda_full_moe_status(
            hipMalloc(&state.device_storage, 2 * 4096 * sizeof(float) + 2 * sizeof(uint32_t)), "allocate scratch")) {
        return false;
    }
    // Keep earlier captured scratch addresses stable when first using FFN.
    if (full_ffn && !state.ffn_prefix_storage && !ggml_cuda_full_moe_status(
            hipMalloc(&state.ffn_prefix_storage, 128 * sizeof(float) + 8 * sizeof(int32_t) +
                      8 * sizeof(float) + sizeof(uint32_t)), "allocate FFN prefix scratch")) {
        return false;
    }
    if (!state.host_status && !ggml_cuda_full_moe_status(
            hipHostMalloc(reinterpret_cast<void **>(&state.host_status), sizeof(uint32_t)), "allocate host status")) {
        return false;
    }
    state.regions.clear();
    for (size_t g = 0; g < count; ++g) {
        ggml_cgraph * graph = graphs[g];
        if (!graph || !graph->uid) {
            return false;
        }
        if (operands[g] && (!operands[g]->data || !operands[g]->buffer ||
            operands[g]->buffer->buft != ggml_backend_cuda_buffer_type(ctx->device))) {
            return false;
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            const ggml_tensor * node = graph->nodes[i];
            // The executor skips empty tensors before invoking any matcher.
            if (ggml_is_empty(node) || node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            if (!full_ffn && (!node->src[0] || node->src[0]->ne[0] != 4096 || node->src[0]->ne[1] != 512)) {
                continue;
            }
            ggml_cuda_full_moe_region region;
            const int start = full_ffn ? i - 14 : i;
            if (!(full_ffn ? ggml_cuda_full_ffn_match(ctx, graph, start, region) :
                            ggml_cuda_full_moe_match(ctx, graph, start, region))) {
                GGML_LOG_ERROR("%s: unsupported full-MoE region %s\n", __func__, node->name);
                return false;
            }
            const size_t type = (region.ffn128 ? region.ffn128_args.down_type : region.args.down_type) == GGML_TYPE_Q6_K ? 1 : 0;
            if (region.ffn128) {
                if (!state.ffn128_queried[type]) {
                    if (!ggml_cuda_full_moe_status(qwen3moe_ffn_router_q8_128_query(
                            &region.ffn128_args, &state.ffn128_query_info[type]), "query FFN128 occupancy")) {
                        return false;
                    }
                    state.ffn128_queried[type] = true;
                }
                region.ffn128_info = state.ffn128_query_info[type];
                if (region.ffn128_info.device != ctx->device) {
                    return false;
                }
            } else if (region.split128) {
                if (!state.split128_queried[type]) {
                    if (!ggml_cuda_full_moe_status(qwen3moe_split_moe_q8_128_query(
                            &region.split128_args, &state.split128_query_info[type]), "query split128 occupancy")) {
                        return false;
                    }
                    state.split128_queried[type] = true;
                }
                region.split128_info = state.split128_query_info[type];
                if (region.split128_info.device != ctx->device) {
                    return false;
                }
            } else if (region.native128) {
                if (!state.native128_queried[type]) {
                    if (!ggml_cuda_full_moe_status(qwen3moe_full_moe_q8_128_query(
                            &region.native128_args, &state.native128_query_info[type]), "query native128 occupancy")) {
                        return false;
                    }
                    state.native128_queried[type] = true;
                }
                region.native128_info = state.native128_query_info[type];
                if (region.native128_info.device != ctx->device) {
                    return false;
                }
            } else if (region.native) {
                if (!state.native_queried[type]) {
                    if (!ggml_cuda_full_moe_status(qwen3moe_full_moe_q8_256_query(
                            &region.native_args, &state.native_query_info[type]), "query native occupancy")) {
                        return false;
                    }
                    state.native_queried[type] = true;
                }
                region.native_info = state.native_query_info[type];
                if (region.native_info.device != ctx->device) {
                    return false;
                }
            } else {
                if (!state.queried[type]) {
                    hipError_t query = qwen3moe_full_moe_query(&region.args, &state.query_info[type]);
                    if (query == hipErrorCooperativeLaunchTooLarge) {
                        region.args.grid_blocks = 64;
                        query = qwen3moe_full_moe_query(&region.args, &state.query_info[type]);
                    }
                    if (!ggml_cuda_full_moe_status(query, "query occupancy")) {
                        return false;
                    }
                    state.queried[type] = true;
                }
                region.info = state.query_info[type];
                region.args.grid_blocks = region.info.grid_blocks;
                if (region.info.device != ctx->device) {
                    return false;
                }
            }
            if (!state.regions.emplace(graph->nodes[start], region).second) {
                return false;
            }
            i += 19;
        }
    }
    if (state.regions.empty()) {
        return false;
    }
    const bool attention_o = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_o_q8_128;
    const bool qkv_prefix = attention_o ||
                           ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_q8_128;
    auto & qkv_state = attention_o ? state.qkv_o : state.qkv;
    if (qkv_prefix && ((state.regions.size() != 93 && state.regions.size() != 94) ||
        !ggml_cuda_qkv_prefix_prepare(ctx, qkv_state, count, graphs, operands, state.error()))) {
        return false;
    }
    if (attention_o && !ggml_cuda_attention_o_prepare(ctx, state.attention_o, qkv_state, count, graphs, operands)) {
        return false;
    }
    *state.host_status = UINT32_MAX;
    state.submitted = true;
    if (!ggml_cuda_full_moe_status(hipMemsetAsync(state.error(), 0, sizeof(uint32_t), state.stream), "reset token status")) {
        state.poisoned = true;
        return false;
    }
    // A prefix error before the first MoE still has a bounded reduction value.
    if (qkv_prefix && !ggml_cuda_full_moe_status(hipMemsetAsync(state.output(), 0, 4096 * sizeof(float), state.stream),
                                               "initialize rejected-token output")) {
        state.poisoned = true;
        return false;
    }
    state.prepared = true;
    return true;
}

static int ggml_cuda_full_moe_dispatch(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int i) {
    auto & state = *ctx->full_moe;
    if (!state.prepared || state.sealed || state.poisoned || ctx->curr_stream_no != 0 || ctx->stream() != state.stream) {
        ctx->execution_extension_failed = true;
        return -1;
    }
    const bool attention_o = ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_o_q8_128;
    if (attention_o || ctx->execution_extension_plan == ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_q8_128) {
        const int prefix_skip = ggml_cuda_qkv_prefix_dispatch(ctx, attention_o ? state.qkv_o : state.qkv, graph, i, state.stream);
        if (prefix_skip < 0) {
            ctx->execution_extension_failed = true;
            state.poisoned = true;
        }
        if (prefix_skip != 0) return prefix_skip;
    }
    if (attention_o) {
        const int output_skip = ggml_cuda_attention_o_dispatch(ctx, state.attention_o, graph, i, state.stream);
        if (output_skip < 0) {
            ctx->execution_extension_failed = true;
            state.poisoned = true;
        }
        if (output_skip != 0) return output_skip;
    }
    const auto it = state.regions.find(graph->nodes[i]);
    if (it == state.regions.end()) {
        return 0;
    }
    const auto & region = it->second;
    const hipError_t launched = region.ffn128 ?
        qwen3moe_ffn_router_q8_128_enqueue(&region.ffn128_args, &region.ffn128_info, state.stream) : region.split128 ?
        qwen3moe_split_moe_q8_128_enqueue(&region.split128_args, &region.split128_info, state.stream) : region.native128 ?
        qwen3moe_full_moe_q8_128_enqueue(&region.native128_args, &region.native128_info, state.stream) : region.native ?
        qwen3moe_full_moe_q8_256_enqueue(&region.native_args, &region.native_info, state.stream) :
        qwen3moe_full_moe_enqueue(&region.args, &region.info, state.stream);
    const float * kernel_output = region.ffn128 ? region.ffn128_args.output : region.split128 ? region.split128_args.output :
                                 region.native128 ? region.native128_args.output :
                                 region.native ? region.native_args.output : region.args.output;
    if (!ggml_cuda_full_moe_status(launched, "enqueue full-MoE") ||
        (!region.ffn_direct && !ggml_cuda_full_moe_status(
            hipMemcpyAsync(region.output->data, kernel_output, 4096 * sizeof(float), hipMemcpyDeviceToDevice, state.stream),
            "copy full-MoE output"))) {
        ctx->execution_extension_failed = true;
        state.poisoned = true;
        return -1;
    }
    ctx->execution_extension_launches++;
    return region.ffn128 ? 33 : 19;
}

static bool ggml_cuda_full_moe_seal(ggml_backend_cuda_context * ctx) {
    auto & state = *ctx->full_moe;
    if (!state.prepared || state.sealed || state.poisoned || ctx->curr_stream_no != 0 || ctx->stream() != state.stream ||
        !ggml_cuda_full_moe_status(hipMemcpyAsync(state.host_status, state.error(), sizeof(uint32_t),
                                               hipMemcpyDeviceToHost, state.stream), "copy token status")) {
        state.poisoned = true;
        return false;
    }
    state.sealed = true;
    return true;
}

static void ggml_cuda_full_moe_cancel(ggml_cuda_full_moe_state & state) {
    state.poisoned = state.poisoned || state.submitted;
    state.prepared = false;
    state.sealed = false;
    // Submitted storage, host status, and captured pointers remain owned.
}

static bool ggml_cuda_full_moe_end(ggml_cuda_full_moe_state & state) {
    const bool success = state.prepared && state.sealed && !state.poisoned && *state.host_status == 0;
    state.poisoned = !success;
    state.prepared = false;
    state.sealed = false;
    state.submitted = false; // The caller has completed its existing synchronization.
    return success;
}

static void ggml_cuda_full_moe_destroy(ggml_backend_cuda_context * ctx) {
    auto * state = ctx->full_moe;
    if (!state) {
        return;
    }
    // Backend free quarantines the entire context before any destructor runs.
    GGML_ASSERT(!state->submitted && !state->poisoned);
#ifdef USE_CUDA_GRAPH
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_full_moe)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_full_moe_q8_256)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_full_moe_q8_128)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_ffn_router_q8_128)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_split_moe_q8_128)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_ffn_router_q8_128_direct)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_q8_128)].clear();
    ctx->cuda_graphs[static_cast<size_t>(ggml_cuda_execution_extension_plan::qwen3moe_split_moe_qkv_prefix_o_q8_128)].clear();
#endif
    if (state->device_storage) {
        (void) ggml_cuda_full_moe_status(hipFree(state->device_storage), "free full-MoE scratch");
    }
    if (state->ffn_prefix_storage) {
        (void) ggml_cuda_full_moe_status(hipFree(state->ffn_prefix_storage), "free FFN prefix scratch");
    }
    if (state->host_status) {
        (void) ggml_cuda_full_moe_status(hipHostFree(state->host_status), "free host status");
    }
    if (state->qkv.scratch) {
        (void) ggml_cuda_full_moe_status(hipFree(state->qkv.scratch), "free QKV prefix scratch");
    }
    if (state->qkv_o.scratch) {
        (void) ggml_cuda_full_moe_status(hipFree(state->qkv_o.scratch), "free QKV/O prefix scratch");
    }
    delete state;
    ctx->full_moe = nullptr;
}
