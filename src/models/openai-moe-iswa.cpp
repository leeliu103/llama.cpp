#include "models.h"

llm_build_openai_moe_iswa::llm_build_openai_moe_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            const llm_proj_group & proj_group = model.layers[il].proj_group(LLM_PROJ_SOURCE_SELF_ATTN);
            const llm_proj_group_output proj = build_proj_group(proj_group, cur, il);

            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_proj_reshape_3d(proj_group, proj, LLM_PROJ_Q, n_rot, n_head,    n_tokens);
            ggml_tensor * Kcur = build_proj_reshape_3d(proj_group, proj, LLM_PROJ_K, n_rot, n_head_kv, n_tokens);
            ggml_tensor * Vcur = build_proj_reshape_3d(proj_group, proj, LLM_PROJ_V, n_rot, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_rot)), il);

            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // MoE branch
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                false, 0.0,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
