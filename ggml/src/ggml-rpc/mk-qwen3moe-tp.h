// The tensor-split Qwen3-235B-A22B decode kernel (qwen3moe_shard.hip built with -DMK_LIBRARY -DMK_LAYOUT_LLAMA
// -DMK_SHARDS=3) as a library for llama.cpp's RPC server: one instance per device of llama.cpp's tensor split
// (-sm tensor over 3 RPC devices), reading that device's slices of llama.cpp's own weights and KV cache.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A tensor slice in device memory as ggml lays it out: ggml type, ne[0] elements per row, ne[1] rows,
// ne[2] matrices (experts).
struct mk_tp_tensor {
    const void * data;
    int32_t      type;
    int64_t      ne[3];
};

// One layer: this device's slices (llama.cpp's split for this layer) and its K / V cache, F16 [cell][head][128]
// with cell i holding position i.
struct mk_tp_layer {
    struct mk_tp_tensor q, k, v, o, gate, up, down;
    struct mk_tp_tensor router, attn_norm, ffn_norm, q_norm, k_norm;
    void * k_cache;
    void * v_cache;
};

struct mk_tp_model {
    int32_t                    rank;      // this device's index in llama.cpp's device list
    int32_t                    n_ranks;   // 3
    int32_t                    n_layers;
    int32_t                    n_cells;   // KV cache cells per layer
    int32_t                    vocab_off; // first vocabulary row of this device's LM head slice
    const struct mk_tp_layer * layers;
    struct mk_tp_tensor        tok_embd;  // whole; data NULL when llama.cpp keeps it on the host (then each
                                          // mk_tp_decode takes the token's row)
    struct mk_tp_tensor        output;    // this device's rows of the LM head
    struct mk_tp_tensor        out_norm;
    const char *               peers;     // the kernel's own exchange: "odl:host:port,..." (transport.h)
};

// NULL with the reason in err when the shapes or types are not the ones the kernel is built for, the GPU cannot
// hold the grid, or the copies do not fit. Copies and re-tiles the weights (the caller keeps its own).
// Connects to the other ranks (every rank calls this at about the same time).
void * mk_tp_create(const struct mk_tp_model * m, char * err, size_t err_len);

// Decodes one token at position pos on this device, the ranks exchanging as the kernel runs (every rank calls
// this with the same token and pos). emb_row: the token's row of tok_embd (host memory), required when
// tok_embd.data was NULL. Writes this device's LM head rows of logits (output.ne[1] floats) to the host buffer
// logits. 0 on success.
int mk_tp_decode(void * mk, int32_t token, int32_t pos, const void * emb_row, float * logits);

void mk_tp_free(void * mk);

#ifdef __cplusplus
}
#endif
