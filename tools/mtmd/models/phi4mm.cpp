#include "models.h"

#include <algorithm>
#include <cstdlib>

static ggml_tensor * phi4mm_view_tokens(ggml_context * ctx, ggml_tensor * x, int64_t token_start, int64_t token_count) {
    GGML_ASSERT(x->ne[0] == 1152);
    GGML_ASSERT(token_start >= 0);
    GGML_ASSERT(token_count > 0);
    GGML_ASSERT(token_start + token_count <= x->ne[1]);
    return ggml_view_2d(ctx, x, x->ne[0], token_count, x->nb[1], token_start * x->nb[1]);
}

static ggml_tensor * phi4mm_concat_tokens(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (a == nullptr) {
        return b;
    }
    return ggml_concat(ctx, a, b, 1);
}

ggml_tensor * clip_graph_phi4mm::pool_crop(ggml_tensor * hidden_states, int crop_idx) {
    GGML_ASSERT(hidden_states->ne[0] == 1152);
    GGML_ASSERT(hidden_states->ne[1] == 32 * 32);
    GGML_ASSERT(crop_idx >= 0 && crop_idx < hidden_states->ne[2]);

    ggml_tensor * crop = ggml_view_2d(
        ctx0, hidden_states, 1152, 32 * 32, hidden_states->nb[1], crop_idx * hidden_states->nb[2]);

    // PyTorch path: [B, 1024, C] -> [B, C, 32, 32] -> AvgPool2d(2, 2)
    crop = ggml_cont(ctx0, ggml_transpose(ctx0, crop));
    crop = ggml_reshape_3d(ctx0, crop, 32, 32, 1152);
    crop = ggml_pool_2d(ctx0, crop, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
    crop = ggml_reshape_2d(ctx0, crop, 16 * 16, 1152);
    crop = ggml_cont(ctx0, ggml_transpose(ctx0, crop));
    return crop;
}

ggml_tensor * clip_graph_phi4mm::build_global_sequence(ggml_tensor * global_pooled) {
    GGML_ASSERT(global_pooled->ne[0] == 1152);
    GGML_ASSERT(global_pooled->ne[1] == 16 * 16);
    GGML_ASSERT(model.image_newline != nullptr);

    ggml_tensor * newline = ggml_reshape_2d(ctx0, model.image_newline, 1152, 1);
    newline = ggml_cast(ctx0, newline, global_pooled->type);

    ggml_tensor * out = nullptr;
    for (int row = 0; row < 16; ++row) {
        ggml_tensor * row_tokens = phi4mm_view_tokens(ctx0, global_pooled, row * 16, 16);
        row_tokens = ggml_concat(ctx0, row_tokens, newline, 1);
        out = phi4mm_concat_tokens(ctx0, out, row_tokens);
    }

    GGML_ASSERT(out != nullptr);
    GGML_ASSERT(out->ne[1] == 16 * 17);
    return out;
}

ggml_tensor * clip_graph_phi4mm::build_sub_sequence(const std::vector<ggml_tensor *> & pooled_crops) {
    const int grid_x = img.phi4mm_grid_x;
    const int grid_y = img.phi4mm_grid_y;
    const int useful_width = img.phi4mm_useful_width;
    const int useful_height = img.phi4mm_useful_height;

    GGML_ASSERT(grid_x > 0 && grid_y > 0);
    GGML_ASSERT((int)pooled_crops.size() == 1 + grid_x * grid_y);
    GGML_ASSERT(useful_width > 0 && useful_width <= grid_x * 16);
    GGML_ASSERT(useful_height > 0 && useful_height <= grid_y * 16);
    GGML_ASSERT(model.image_newline != nullptr);

    ggml_tensor * newline = ggml_reshape_2d(ctx0, model.image_newline, 1152, 1);
    newline = ggml_cast(ctx0, newline, pooled_crops[0]->type);

    ggml_tensor * out = nullptr;
    for (int y = 0; y < useful_height; ++y) {
        const int tile_y = y / 16;
        const int local_y = y % 16;
        ggml_tensor * row = nullptr;

        for (int tile_x = 0; tile_x < grid_x; ++tile_x) {
            const int global_x0 = tile_x * 16;
            const int global_x1 = global_x0 + 16;
            if (global_x0 >= useful_width) {
                break;
            }

            const int local_x0 = 0;
            const int local_x1 = std::min(global_x1, useful_width) - global_x0;
            GGML_ASSERT(local_x1 > local_x0);

            const int crop_idx = 1 + tile_y * grid_x + tile_x;
            ggml_tensor * segment = phi4mm_view_tokens(
                ctx0, pooled_crops[crop_idx], local_y * 16 + local_x0, local_x1 - local_x0);
            row = phi4mm_concat_tokens(ctx0, row, segment);
        }

        GGML_ASSERT(row != nullptr);
        row = ggml_concat(ctx0, row, newline, 1);
        out = phi4mm_concat_tokens(ctx0, out, row);
    }

    GGML_ASSERT(out != nullptr);
    GGML_ASSERT(out->ne[1] == useful_height * (useful_width + 1));
    return out;
}

ggml_tensor * clip_graph_phi4mm::build_projector(ggml_tensor * image_tokens) {
    GGML_ASSERT(model.mm_0_w != nullptr);
    GGML_ASSERT(model.mm_0_b != nullptr);
    GGML_ASSERT(model.mm_2_w != nullptr);
    GGML_ASSERT(model.mm_2_b != nullptr);

    ggml_tensor * cur = build_mm(model.mm_0_w, image_tokens);
    cur = ggml_add(ctx0, cur, model.mm_0_b);
    // Official Phi4MMImageEmbedding uses nn.GELU() for img_projection.
    cur = ggml_gelu_erf(ctx0, cur);
    cur = build_mm(model.mm_2_w, cur);
    cur = ggml_add(ctx0, cur, model.mm_2_b);
    return cur;
}

ggml_cgraph * clip_graph_phi4mm::build() {
    GGML_ASSERT(proj_type == PROJECTOR_TYPE_PHI4MM);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(n_patches == 32 * 32);
    GGML_ASSERT(img.phi4mm_grid_x > 0);
    GGML_ASSERT(img.phi4mm_grid_y > 0);
    GGML_ASSERT(n_batch == 1 + img.phi4mm_grid_x * img.phi4mm_grid_y);

    ggml_tensor * inp = build_inp();

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches * n_batch);
    ggml_set_name(positions, "phi4mm_positions");
    ggml_set_input(positions);

    ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);
    learned_pos_embd = ggml_reshape_3d(ctx0, learned_pos_embd, n_embd, n_patches, n_batch);

    ggml_tensor * attn_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_patches, n_patches, 1, n_batch);
    ggml_set_name(attn_mask, "phi4mm_attn_mask");
    ggml_set_input(attn_mask);

    build_vit_opts opts;
    opts.attn_mask = attn_mask;

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr,
                            opts);

    cb(cur, "phi4mm_hidden_states_minus2", -1);

    const char * hidden_dump_dir = std::getenv("MTMD_DEBUG_PHI4MM_HIDDEN_STATES_DUMP");
    if (hidden_dump_dir != nullptr && hidden_dump_dir[0] != '\0') {
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    std::vector<ggml_tensor *> pooled_crops;
    pooled_crops.reserve(n_batch);
    for (int crop_idx = 0; crop_idx < n_batch; ++crop_idx) {
        pooled_crops.push_back(pool_crop(cur, crop_idx));
    }

    ggml_tensor * sub_img = build_sub_sequence(pooled_crops);
    ggml_tensor * glb_img = build_global_sequence(pooled_crops[0]);

    ggml_tensor * view_sep = ggml_reshape_2d(ctx0, model.view_seperator, 1152, 1);
    view_sep = ggml_cast(ctx0, view_sep, sub_img->type);

    // Official config uses hd_transform_order = "sub_glb".
    cur = ggml_concat(ctx0, sub_img, view_sep, 1);
    cur = ggml_concat(ctx0, cur, glb_img, 1);
    cur = ggml_cont_2d(ctx0, cur, 1152, img.phi4mm_num_img_tokens);
    cb(cur, "phi4mm_hd_tokens", -1);

    GGML_ASSERT(cur->ne[1] == img.phi4mm_num_img_tokens);

    cur = build_projector(cur);
    cb(cur, "phi4mm_projected_embeddings", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
