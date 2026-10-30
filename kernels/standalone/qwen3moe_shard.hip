// One shard of a tensor-split persistent decode kernel for Qwen3 MoE Q4_K_M (Qwen3-235B-A22B or
// Qwen3-30B-A3B): every layer is split over kShards boxes, each running this kernel on its slice and
// exchanging the 16 KB partial hidden vector twice per layer through pinned host memory and a host
// proxy (cluster/transport.h). Same stage structure and GEMV machinery as qwen3moe_decode.hip.
//
// Shard r of N holds: Q heads r * H / N .. and KV heads r * KV / N .. (attention needs no exchange);
// the output projection's input columns of those heads (a partial of every row of x); rows
// r * F / N .. of every expert's gate and up and the matching input columns of its down (a partial
// of every row); vocabulary rows r * V / N .. of the LM head (an argmax exchange). The router (F32),
// norms and embedding are on every shard, so every shard holds the same x.
//
// Per layer, with exchanges (N > 1):
//   qkv     RMSNorm + int8 quant of x; this shard's Q, K, V rows
//   attn    per (8 query heads of a KV head, context chunk): head norm + RoPE, cache, softmax, P.V
//   o       this shard's columns of O for every row: partial -> host (write-through stores)
//   [exchange: sum of the shards' partials, in rank order]
//   xup1    x += sum; router partial logits over this block's rows of x
//   gate_up route (top-8 of 128); this shard's gate / up rows of the 8 experts, act = silu(g) * u
//   down    this shard's columns of the 8 experts' down, routing-weighted sum: partial -> host
//   [exchange]
//   xup2    x += sum
// With N = 1 the o and down stages add to x directly and o computes the router partials, as in
// qwen3moe_decode.hip.
//
// An expert's down slice that does not start or end on a superblock (F / N not a multiple of 256)
// keeps whole superblocks with the scales outside the slice set to zero, and its activations at their
// positions in those superblocks (zeros elsewhere): exact, at the cost of reading the padding.
//
//   hipcc -O3 -std=c++17 --offload-arch=<gfx> -DMK_MODEL_235B -DMK_SHARDS=2 -o shard kernels/standalone/qwen3moe_shard.hip
//   shard check                                   # random 2-layer model, N shards in this process, vs a CPU copy
//   shard bench <part-00001.gguf> [layers]        # N shards in this process (GPU r % devices), real exchanges
//   shard compare <gguf> <ref_prefix> [layers]    # teacher-forced against tools/llama_ref.cpp
//   shard run <gguf> <rank> <host:port,...> [layers] [tokens]   # one shard per process / box, TCP
// MK_STAGE_SPLIT=1: finish each attention/MoE kernel before a host exchange (no device mailbox wait).
// MK_SOLO=1 with bench: rank 0 alone, each exchange returning its own partial (timing of one shard).

#include "common.h"
#include "../../cluster/transport.h"
#ifdef MK_LIBRARY
#include "mk_qwen3moe_tp.h"
// Every allocation of a library instance, so that mk_tp_free releases it (llama.cpp creates a kernel per context).
static std::vector<std::pair<void*, bool>>* g_allocs = nullptr;  // (pointer, pinned host)
template <class T>
static hipError_t mk_malloc(T** p, size_t n) {
  const hipError_t e = (hipMalloc)((void**)p, n);
  if (e == hipSuccess && g_allocs) g_allocs->push_back({(void*)*p, false});
  return e;
}
template <class T>
static hipError_t mk_host_malloc(T** p, size_t n, unsigned flags) {
  const hipError_t e = (hipHostMalloc)((void**)p, n, flags);
  if (e == hipSuccess && g_allocs) g_allocs->push_back({(void*)*p, true});
  return e;
}
#define hipMalloc mk_malloc
#define hipHostMalloc mk_host_malloc
#endif

#include <chrono>
#include <memory>
#include <thread>
#include <utility>

namespace mk_shard {

#if defined(MK_MODEL_30B)
constexpr int kHidden = 2048, kHeads = 32, kFFE = 768;
constexpr const char* kModelName = "Qwen3-30B-A3B";
#else
constexpr int kHidden = 4096, kHeads = 64, kFFE = 1536;
constexpr const char* kModelName = "Qwen3-235B-A22B";
#endif
constexpr int kKV = 4;
constexpr int kDim = 128;
constexpr int kRot = kDim / 2;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr float kEps = 1e-6f;
constexpr float kTheta = 1000000.f;
constexpr int kBlock = 256;
constexpr int kWaves = kBlock / 32;
#ifndef MK_SHARDS
#define MK_SHARDS 2
#endif
#ifndef MK_GRID
#define MK_GRID 32
#endif
#ifndef MK_STAGE_SPLIT
#define MK_STAGE_SPLIT 0
#endif
constexpr int kShards = MK_SHARDS;
#ifndef MK_DEPTH
#define MK_DEPTH 4
#endif
constexpr int kD = MK_DEPTH;  // 16-byte chunks in flight per lane in the weight streams
constexpr int kGrid = MK_GRID;
constexpr int kQPK = kHeads / kKV;          // query heads per KV head
constexpr int kIG = kQPK / 8;               // attention items of 8 query heads per KV head
constexpr int kNG = kHeads / 8;             // groups of 8 query heads in the model
// Shard r holds groups grp_first(r) .. grp_first(r + 1) - 1 and the KV heads they use (with 3 shards a KV
// head can be on two of them). Every shard is laid out for the largest share; the others' last groups
// are zero weights and their attention items are skipped.
// MK_LAYOUT_LLAMA: llama.cpp's tensor split (-sm tensor) on 3 devices instead. A layer's KV heads split 1 / 1 / 2
// (query heads 16 / 16 / 32), the sizes rotating with the layer: device (j + layer) % 3 gets the j-th size, and the
// slices follow the device order along the tensor. A shard's role in a layer is the index of its size (role 2: two
// KV heads); its heads start where the earlier devices' end (llama_grp_first). Experts: device d rows 512 d ...
#ifdef MK_LAYOUT_LLAMA
static_assert(kShards == 3 && kHeads == 64 && kKV == 4, "llama.cpp's split as laid out here: 235B on 3 devices");
__host__ __device__ constexpr int grp_first(int r) { return r < kShards ? 2 * r : kNG; }
__host__ __device__ constexpr int layer_role(int rank, int layer) { return (rank + kShards - layer % kShards) % kShards; }
// the first head group of device rank's slice in a layer
constexpr int llama_grp_first(int rank, int layer) {
  int g = 0;
  for (int d = 0; d < rank; d++) g += grp_first(layer_role(d, layer) + 1) - grp_first(layer_role(d, layer));
  return g;
}
#else
__host__ __device__ constexpr int grp_first(int r) { return (r * kNG + kShards - 1) / kShards; }
__host__ __device__ constexpr int layer_role(int rank, int) { return rank; }
#endif
__host__ __device__ constexpr int grp_count(int r) { return grp_first(r + 1) - grp_first(r); }
__host__ __device__ constexpr int kv_first(int r) { return grp_first(r) / kIG; }
__host__ __device__ constexpr int kv_count(int r) { return (grp_first(r + 1) - 1) / kIG - kv_first(r) + 1; }
constexpr int max_groups() {
  int m = 0;
  for (int r = 0; r < kShards; r++) m = grp_count(r) > m ? grp_count(r) : m;
  return m;
}
constexpr int max_kv() {
  int m = 0;
  for (int r = 0; r < kShards; r++) m = kv_count(r) > m ? kv_count(r) : m;
  return m;
}
constexpr int kGroups = max_groups();  // attention head groups of a shard
constexpr int kKVS = max_kv();         // and KV heads
// KV heads per cache position of a layer in which the shard has this role: llama.cpp's cache holds only its own.
#ifdef MK_LAYOUT_LLAMA
__host__ __device__ constexpr int kv_stride(int role) { return kv_count(role); }
#else
__host__ __device__ constexpr int kv_stride(int) { return kKVS; }
#endif
// Groups per attention item from kSeqTokens on (they share the K/V chunk): the groups of a KV head.
constexpr int kSet = kGroups % kIG == 0 ? kIG : 1;
constexpr int kHS = kGroups * 8;            // this shard's query heads
constexpr int kQS = kHS * kDim;             // Q rows of this shard
constexpr int kKVR = kKVS * kDim;           // K (and V) rows
constexpr int kHSB = kHidden / 256;         // superblocks of a row of K = hidden
constexpr int kSPW = kHSB / kWaves;         // of them per wave
constexpr int kQSB = kQS / 256;             // superblocks of an O row (this shard's columns)
constexpr int kHB = kHidden / 32;           // row blocks of x
constexpr int kOR = kHB / kGrid;            // of them per block (o, xup, down)
// o: the waves form kORS sets of kOWR; set s takes row blocks s kORW .., wave i of a set superblocks
// i, i + kOWR, ... (kOWR = 8 unless the slice's superblocks do not divide over 8 waves: 3 shards).
constexpr int o_waves_per_row() {
  int d = kWaves;
  while (kQSB % d || kWaves % d) d--;
  return d;
}
constexpr int kOWR = o_waves_per_row();
constexpr int kORS = kWaves / kOWR;
constexpr int kORW = kOR / kORS;
constexpr int kOSW = kQSB / kOWR;
constexpr int kOQG = (kQSB + kWaves - 1) / kWaves;  // attention-output groups per wave (zero-padded)
constexpr int kFFS = kFFE / kShards;        // expert rows of this shard
constexpr int kRBE = kFFS / 32;             // row blocks per expert
constexpr int kPairs = kTopK * kRBE;        // gate / up row-block pairs per layer
constexpr int kPPB = kPairs / kGrid;        // per block
constexpr int kPPC = kPPB * 2 * kSPW <= 12 ? kPPB : kPPB / 2;  // per stream
constexpr int kNQK = (kQS + kKVR) / 32;     // Q and K row blocks
constexpr int kNV = kKVR / 32;              // V row blocks
// qkv balance, a V row block counting as 1.5 (Q6_K in half the layers): the last kNV blocks take one V row
// block and kVQ Q / K row blocks each, the others the remaining Q / K row blocks, kQLo or kQLo + 1 each.
constexpr int kQT = (2 * kNQK + 3 * kNV + 2 * kGrid - 1) / (2 * kGrid);
constexpr int kVQ = kQT >= 2 ? kQT - 2 : 0;
constexpr int kQRest = kNQK - kNV * kVQ, kQBlocks = kGrid - kNV;
constexpr int kQLo = kQRest / kQBlocks, kQExtra = kQRest % kQBlocks;

__host__ __device__ constexpr int down_first_sb(int r) { return r * kFFS / 256; }
__host__ __device__ constexpr int down_sbs(int r) { return ((r + 1) * kFFS + 255) / 256 - down_first_sb(r); }
constexpr int down_sbs_max() {
  int m = 0;
  for (int r = 0; r < kShards; r++) m = down_sbs(r) > m ? down_sbs(r) : m;
  return m;
}
constexpr bool down_sbs_equal() {
  for (int r = 0; r < kShards; r++)
    if (down_sbs(r) != down_sbs_max()) return false;
  return true;
}
constexpr int kDSB = down_sbs_max();        // superblocks of a down row of this shard
constexpr int kFFP = kDSB * 256;            // padded activations per expert
// Down row blocks per stream: long Q6_K streams make the compiler spill.
constexpr int down_rows_per_stream() {
  int d = kOR;
  while (d > 1 && (d * kDSB > 8 || kOR % d)) d--;
  return d;
}
constexpr int kDH = down_rows_per_stream();
constexpr int cmax(int a, int b) { return a > b ? a : b; }
constexpr int kXQ = cmax(cmax(kTopK * kFFP, kHidden), kOQG * kWaves * 256);

static_assert(kQPK % 8 == 0 && kNG >= kShards, "items hold 8 query heads, at least one group per shard");
static_assert(kHSB % kWaves == 0, "K = hidden splits over the 8 waves");
static_assert(kWaves % kOWR == 0 && kOR % kORS == 0 && kORW * kOSW <= 12, "o: wave sets over the row blocks");
static_assert(kFFS % 128 == 0 && down_sbs_equal(), "down slices of whole Q6_K halves, the same count on every shard");
static_assert(kHB % kGrid == 0 && kPairs % kGrid == 0 && kPPB % kPPC == 0, "stages divide evenly over the grid");
static_assert(kNV <= kGrid && kTopK == kWaves && kExperts == 128, "V items, down slots and route() assume these");

constexpr int kQ4Chunks = 9;   // block_q4_K: {d, dmin, scales[12]}, qs[128]
constexpr int kQ6Chunks = 13;  // block_q6_K minus d: ql[128], qh[64], scales[16]
constexpr int kQ4Bytes = 144;
constexpr int kQ6Bytes = 210;
constexpr int kMaxChunk = 512;  // context tokens per attention work item
constexpr int kMaxSplits = 64;  // context chunks per head group
constexpr unsigned kOpSum = 0, kOpArgmax = 1;
constexpr int kSeqTokens = 2048;  // context from which an attention item does all head groups of its KV head
__host__ __device__ constexpr int attn_seq(int ntok) { return ntok >= kSeqTokens ? kSet : 1; }

struct Layer {
  const u4v* q;
  const u4v* k;
  const u4v* v;
  const uint16_t* v_d;
  const u4v* o;
  const float* router;  // [kExperts][kHidden]
  const u4v* gate;      // [kExperts * kFFS] x kHidden
  const u4v* up;
  const u4v* down;  // [kExperts * kHidden] x kFFP
  const uint16_t* down_d;
  int down_q6;
  const float* attn_norm;
  const float* ffn_norm;
  const float* q_norm;
  const float* k_norm;
};

// Pinned host memory shared with this shard's proxy thread.
struct Xchg {
  alignas(256) unsigned out_seq;  // GPU: exchange number out_seq is in out[]
  unsigned op, n;                 // kOpSum: n floats; kOpArgmax: {value, index}
  alignas(256) unsigned in_seq;   // host: its result is in in[]
  alignas(256) float out[kHidden];
  alignas(256) float in[kHidden];
};

struct Args {
  int n_layers;
  int vocab;      // this shard's LM head rows
  int vocab_off;  // index of its first row
  int act_off;    // position of this shard's first expert row in its padded down slice
  int rank;
  int pos;
  int n_steps;
  int max_ctx;
  unsigned gen_base;
  unsigned xseq_base;
  const Layer* layers;
  const uint8_t* emb;
  const u4v* lm;
  const uint16_t* lm_d;
  const float* out_norm;
  float* x;
  float* q;
  float* k;
  float* v;
  float* router_part;  // [kExperts][kHB]
  float* router_ss;    // [kHB]
  float* act;          // [kTopK][kFFP], zero outside this shard's rows
  float* logits;
  float* part_acc;  // [kHS][kMaxSplits][kDim]
  float* part_m;    // [kHS][kMaxSplits]
  float* part_l;
  float* attn;         // [kOQG * kWaves * 256], zero past this shard's heads
  unsigned* attn_cnt;  // [kGroups]
  uint16_t* const* kcache;  // [n_layers]; position p, KV head g: + (p * kKVS + g) * kDim
  uint16_t* const* vcache;
  float* trace;      // [n_layers + 1][kHidden], null to skip
  int* trace_sel;    // [n_layers][kTopK], null to skip
  float* best_v;
  int* best_i;
  int* token;
  int* tokens_out;
  const int* inputs;
  float* logits_all;  // [n_steps][vocab]
  unsigned* bar_count;
  unsigned long long* stamps;
  unsigned* status;
  Xchg* xc;  // null with one shard
};

struct Smem {
  int8_t xq[kXQ] __attribute__((aligned(16)));
  int xs16[kXQ / 16];
  float xd[kXQ / 256];
  float red[32 * kWaves];
  float q[8][kDim];
  float cs[kRot];
  float sn[kRot];
  union {
    float p[8][kMaxChunk];
    float vred[kWaves / 2][8][kDim];  // half the waves, so that two blocks fit in Strix Halo's 64 KB per WGP
    float part[kWaves * 12][32];
  };
  uint16_t kvnew[2][kDim] __attribute__((aligned(16)));
  float yred[kWaves][32];
  int best_i[kWaves];
  int flag;
  int sel[kTopK];
  float selw[kTopK];
  float pk[kTopK];
  unsigned long long rk[kExperts] __attribute__((aligned(16)));
  float xw[32 * kOR];
};

__device__ __forceinline__ void grid_barrier(const Args& a, unsigned& my_gen,
                                             unsigned long long timeout = kBarrierTimeout) {
  ::grid_barrier(a.bar_count, my_gen, timeout);
}

// ------------------------------------------------------------- block helpers
// Activations in groups of 256 (one int8 scale each). Wave w owns groups w, w + 8, ...; lane l holds
// the 8 consecutive values 256 * group + 8 * l.
template <int G>
__device__ void load_groups(const float* __restrict__ src, float (&v)[G][8]) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < G; g++) {
    const float4* p = (const float4*)(src + (wv + kWaves * g) * 256 + 8 * lane);
    const float4 a = p[0], b = p[1];
    v[g][0] = a.x, v[g][1] = a.y, v[g][2] = a.z, v[g][3] = a.w;
    v[g][4] = b.x, v[g][5] = b.y, v[g][6] = b.z, v[g][7] = b.w;
  }
}

template <int G>
__device__ void groups_rmsnorm(Smem& sm, float (&v)[G][8], const float (&wg)[G][8]) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
  float ss = 0.f;
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) ss = __builtin_fmaf(v[g][e], v[g][e], ss);
  ss = wave_sum(ss);
  lds_sync();
  if (lane == 0) sm.red[wv] = ss;
  lds_sync();
  float tot = 0.f;
#pragma unroll
  for (int k = 0; k < kWaves; k++) tot += sm.red[k];
  const float inv = 1.f / sqrtf(tot / (float)(G * kBlock * 8) + kEps);
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) v[g][e] = v[g][e] * inv * wg[g][e];
}

// v / d with a wave-uniform d: refined reciprocal and one fma correction (Markstein).
__device__ __forceinline__ float uniform_rcp(float d) {
  const float r = __builtin_amdgcn_rcpf(d);
  return fmaf(fmaf(-d, r, 1.f), r, r);
}
__device__ __forceinline__ float div_by(float v, float d, float r) {
  const float q = v * r;
  return fmaf(fmaf(-q, d, v), r, q);
}

__device__ __forceinline__ void quantize_group(Smem& sm, int grp, const float (&v)[8]) {
  const int lane = threadIdx.x & 31;
  float m = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) m = fmaxf(m, fabsf(v[e]));
  m = wave_max(m);
  const float scale = m > 0.f ? m / 127.f : 1.f, rs = uniform_rcp(scale);
  unsigned packed[2] = {0, 0};
  int s8 = 0;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int q = (int)rintf(div_by(v[e], scale, rs));
    q = q > 127 ? 127 : q < -127 ? -127 : q;
    packed[e >> 2] |= ((unsigned)q & 0xFFu) << (8 * (e & 3));
    s8 += q;
  }
  *(uint2*)(sm.xq + grp * 256 + 8 * lane) = make_uint2(packed[0], packed[1]);
  const int s16 = s8 + __shfl_xor(s8, 1);
  if ((lane & 1) == 0) sm.xs16[grp * 16 + (lane >> 1)] = s16;
  if (lane == 0) sm.xd[grp] = scale;
}

template <int G>
__device__ void groups_quantize(Smem& sm, const float (&v)[G][8]) {
  const int wv = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < G; g++) quantize_group(sm, wv + kWaves * g, v[g]);
  lds_sync();
}

// Weight pointers come from the layer table, so the compiler would emit flat loads (which also count
// in DScnt); these are global loads.
typedef const u4v __attribute__((address_space(1)))* gu4v_ptr;
typedef const uint16_t __attribute__((address_space(1)))* gu16_ptr;
__device__ __forceinline__ u4v ldg(const u4v* p) { return *(gu4v_ptr)p; }
__device__ __forceinline__ uint16_t ldg(const uint16_t* p) { return *(gu16_ptr)p; }

// ------------------------------------------------------------------- GEMV
//   Q4_K tile: [M/32][K/256][9 chunks][32 lanes][16 B]  (chunk 0 = d, dmin, scales; 1-8 = qs)
//   Q6_K tile: [M/32][K/256][13 chunks][32 lanes][16 B] (ql, qh, scales) + d as [M/32][K/256][32] fp16
struct B6 {
  u4v c[kQ6Chunks];
  uint16_t d;
};

__device__ __forceinline__ void fetch_q6(B6& b, const u4v* __restrict__ W, const uint16_t* __restrict__ D, int rb,
                                         int SB, int sb) {
  const unsigned blk = rb * SB + sb;
  const u4v* p = W + (blk * kQ6Chunks * 32 + (threadIdx.x & 31));
#pragma unroll
  for (int c = 0; c < kQ6Chunks; c++) b.c[c] = ldg(p + c * 32);
  b.d = ldg(D + blk * 32 + (threadIdx.x & 31));
}

__device__ __forceinline__ float dot_q6(const B6& b, int sb, const Smem& sm, float acc) {
  const int8_t* xq = sm.xq + sb * 256;
  int sumi[16];
#pragma unroll
  for (int s = 0; s < 16; s++) sumi[s] = 0;
#pragma unroll
  for (int h = 0; h < 2; h++) {
#pragma unroll
    for (int g = 0; g < 2; g++) {
      const u4v A = b.c[4 * h + g], B = b.c[4 * h + 2 + g], H = b.c[8 + 2 * h + g];
      const u4v x1 = *(const u4v*)(xq + 128 * h + 16 * g);
      const u4v x2 = *(const u4v*)(xq + 128 * h + 16 * g + 32);
      const u4v x3 = *(const u4v*)(xq + 128 * h + 16 * g + 64);
      const u4v x4 = *(const u4v*)(xq + 128 * h + 16 * g + 96);
      const unsigned av[4] = {A.x, A.y, A.z, A.w}, bv[4] = {B.x, B.y, B.z, B.w}, hv[4] = {H.x, H.y, H.z, H.w};
      const unsigned v1[4] = {x1.x, x1.y, x1.z, x1.w}, v2[4] = {x2.x, x2.y, x2.z, x2.w};
      const unsigned v3[4] = {x3.x, x3.y, x3.z, x3.w}, v4[4] = {x4.x, x4.y, x4.z, x4.w};
      const int s = 8 * h + g;
#pragma unroll
      for (int d = 0; d < 4; d++) {
        const unsigned qa = av[d], qb = bv[d], qh = hv[d];
        const int q1 = (int)((qa & 0x0F0F0F0Fu) | ((qh << 4) & 0x30303030u));
        const int q2 = (int)((qb & 0x0F0F0F0Fu) | ((qh << 2) & 0x30303030u));
        const int q3 = (int)(((qa >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u));
        const int q4 = (int)(((qb >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u));
        sumi[s] = __builtin_amdgcn_sudot4(false, q1, true, (int)v1[d], sumi[s], false);
        sumi[s + 2] = __builtin_amdgcn_sudot4(false, q2, true, (int)v2[d], sumi[s + 2], false);
        sumi[s + 4] = __builtin_amdgcn_sudot4(false, q3, true, (int)v3[d], sumi[s + 4], false);
        sumi[s + 6] = __builtin_amdgcn_sudot4(false, q4, true, (int)v4[d], sumi[s + 6], false);
      }
    }
  }
  const unsigned scw[4] = {b.c[12].x, b.c[12].y, b.c[12].z, b.c[12].w};
  const int* xs = sm.xs16 + sb * 16;
  int isum = 0;
#pragma unroll
  for (int s = 0; s < 16; s++) {
    const int sc = (int)(int8_t)(scw[s >> 2] >> (8 * (s & 3)));
    isum += __mul24(sc, sumi[s] - 32 * xs[s]);
  }
  return __builtin_fmaf(sm.xd[sb], f16_to_f32(b.d) * (float)isum, acc);
}

struct QMat {
  const u4v* w;
  const uint16_t* d;  // Q6_K scales, null for Q4_K
  int SB;             // superblocks per row
};

__device__ __forceinline__ float lm_row_dot(const QMat& m, int rb, const Smem& sm, int sb0, float acc) {
#pragma unroll 1
  for (int sb = sb0; sb < m.SB; sb++) {
    B6 b;
    fetch_q6(b, m.w, m.d, rb, m.SB, sb);
    acc = dot_q6(b, sb, sm, acc);
  }
  return acc;
}

// Chunk-pipelined row dots: a wave's NB weight blocks are one stream of 16-byte chunks per lane with D
// in flight. Block b is Q6_K if bit b of Q6 is set; it accumulates onto block b - 1 if bit b of CH is
// set (one row over several superblocks, fused multiply-add as in the whole-block dots).
__host__ __device__ constexpr int blk_chunks(unsigned q6, int b) { return (q6 >> b) & 1 ? kQ6Chunks : kQ4Chunks; }
__host__ __device__ constexpr int q6_chunk(int pos) {
  constexpr int order[kQ6Chunks] = {0, 2, 8, 1, 3, 9, 4, 6, 10, 5, 7, 11, 12};
  return order[pos];
}
// Bits of the blocks after the first in each group of n: groups of n consecutive blocks chain.
__host__ __device__ constexpr unsigned chain_mask(int groups, int n) {
  unsigned m = 0;
  for (int g = 0; g < groups; g++)
    for (int i = 1; i < n; i++) m |= 1u << (g * n + i);
  return m;
}

template <int NB, int D, unsigned Q6, unsigned CH>
struct Stream {
  static constexpr int total() {
    int n = 0;
    for (int b = 0; b < NB; b++) n += blk_chunks(Q6, b);
    return n;
  }
  static constexpr int N = total();
  static constexpr int blk_of(int k) {
    int b = 0;
    while (k >= blk_chunks(Q6, b)) k -= blk_chunks(Q6, b++);
    return b;
  }
  static constexpr int pos_of(int k) {
    int b = 0;
    while (k >= blk_chunks(Q6, b)) k -= blk_chunks(Q6, b++);
    return k;
  }
  static constexpr int chunk_of(int k) { return (Q6 >> blk_of(k)) & 1 ? q6_chunk(pos_of(k)) : pos_of(k); }

  const u4v* base[NB];
  int sb[NB];
  uint16_t d6[NB];
  const uint16_t* dptr[NB];
  u4v buf[D];

  template <int... K>
  __device__ __forceinline__ void pre_k(std::integer_sequence<int, K...>) {
    ((buf[K] = ldg(base[blk_of(K)] + chunk_of(K) * 32)), ...);
  }
  template <int... B>
  __device__ __forceinline__ void pre_d(std::integer_sequence<int, B...>) {
    ((((Q6 >> B) & 1) ? (void)(d6[B] = ldg(dptr[B])) : (void)0), ...);
  }
  __device__ __forceinline__ void pre() {
    pre_d(std::make_integer_sequence<int, NB>{});
    pre_k(std::make_integer_sequence<int, (D < N ? D : N)>{});
  }

  template <int k>
  __device__ __forceinline__ void step(const Smem& sm, float (&out)[NB], int (&sumi)[16], u4v& hdr, u4v& qa, u4v& qb,
                                       float& acc) {
    constexpr int b = blk_of(k), pos = pos_of(k);
    constexpr bool q6 = (Q6 >> b) & 1, chain = (CH >> b) & 1;
    const int8_t* xq = sm.xq + sb[b] * 256;
    const int* xs = sm.xs16 + sb[b] * 16;
    const u4v cur = buf[k % D];
    if constexpr (k + D < N) buf[k % D] = ldg(base[blk_of(k + D)] + chunk_of(k + D) * 32);
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (!q6) {
      if constexpr (pos == 0) {
        hdr = cur;
#pragma unroll
        for (int s = 0; s < 8; s++) sumi[s] = 0;
      } else {
        const int cc = pos - 1, j = cc >> 1, l0 = 16 * (cc & 1);
        const u4v xl = *(const u4v*)(xq + 64 * j + l0);
        const u4v xh = *(const u4v*)(xq + 64 * j + 32 + l0);
        const unsigned qd[4] = {cur.x, cur.y, cur.z, cur.w};
        const unsigned xa[4] = {xl.x, xl.y, xl.z, xl.w};
        const unsigned xb[4] = {xh.x, xh.y, xh.z, xh.w};
#pragma unroll
        for (int d = 0; d < 4; d++) {
          const int lo = (int)(qd[d] & 0x0F0F0F0Fu);
          const int hi = (int)((qd[d] >> 4) & 0x0F0F0F0Fu);
          sumi[2 * j] = __builtin_amdgcn_sudot4(false, lo, true, (int)xa[d], sumi[2 * j], false);
          sumi[2 * j + 1] = __builtin_amdgcn_sudot4(false, hi, true, (int)xb[d], sumi[2 * j + 1], false);
        }
      }
      if constexpr (pos == kQ4Chunks - 1) {
        const unsigned s0 = hdr.y, s1 = hdr.z, s2 = hdr.w;
        int isc = 0, imn = 0;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const int sc_lo = (s0 >> (8 * t)) & 63, mn_lo = (s1 >> (8 * t)) & 63;
          const unsigned b8 = (s2 >> (8 * t)) & 0xFF;
          const int sc_hi = (b8 & 0xF) | (((s0 >> (8 * t + 6)) & 3) << 4);
          const int mn_hi = (b8 >> 4) | (((s1 >> (8 * t + 6)) & 3) << 4);
          isc += __mul24(sc_lo, sumi[t]);
          imn += __mul24(mn_lo, xs[2 * t] + xs[2 * t + 1]);
          isc += __mul24(sc_hi, sumi[t + 4]);
          imn += __mul24(mn_hi, xs[2 * (t + 4)] + xs[2 * (t + 4) + 1]);
        }
        const float d = half_lo(hdr.x), dmin = half_hi(hdr.x);
        acc = __builtin_fmaf(sm.xd[sb[b]], __builtin_fmaf(d, (float)isc, -(dmin * (float)imn)), chain ? acc : 0.f);
        out[b] = acc;
      }
    } else {
      if constexpr (pos == 0) {
#pragma unroll
        for (int s = 0; s < 16; s++) sumi[s] = 0;
      }
      if constexpr (pos < 12) {
        const int grp = pos / 3, h = grp >> 1, g = grp & 1;
        if constexpr (pos % 3 == 0) qa = cur;
        if constexpr (pos % 3 == 1) qb = cur;
        if constexpr (pos % 3 == 2) {
          const u4v H = cur;
          const u4v x1 = *(const u4v*)(xq + 128 * h + 16 * g);
          const u4v x2 = *(const u4v*)(xq + 128 * h + 16 * g + 32);
          const u4v x3 = *(const u4v*)(xq + 128 * h + 16 * g + 64);
          const u4v x4 = *(const u4v*)(xq + 128 * h + 16 * g + 96);
          const unsigned av[4] = {qa.x, qa.y, qa.z, qa.w}, bv[4] = {qb.x, qb.y, qb.z, qb.w};
          const unsigned hv[4] = {H.x, H.y, H.z, H.w};
          const unsigned v1[4] = {x1.x, x1.y, x1.z, x1.w}, v2[4] = {x2.x, x2.y, x2.z, x2.w};
          const unsigned v3[4] = {x3.x, x3.y, x3.z, x3.w}, v4[4] = {x4.x, x4.y, x4.z, x4.w};
          const int s = 8 * h + g;
#pragma unroll
          for (int d = 0; d < 4; d++) {
            const unsigned a_ = av[d], b_ = bv[d], h_ = hv[d];
            const int q1 = (int)((a_ & 0x0F0F0F0Fu) | ((h_ << 4) & 0x30303030u));
            const int q2 = (int)((b_ & 0x0F0F0F0Fu) | ((h_ << 2) & 0x30303030u));
            const int q3 = (int)(((a_ >> 4) & 0x0F0F0F0Fu) | (h_ & 0x30303030u));
            const int q4 = (int)(((b_ >> 4) & 0x0F0F0F0Fu) | ((h_ >> 2) & 0x30303030u));
            sumi[s] = __builtin_amdgcn_sudot4(false, q1, true, (int)v1[d], sumi[s], false);
            sumi[s + 2] = __builtin_amdgcn_sudot4(false, q2, true, (int)v2[d], sumi[s + 2], false);
            sumi[s + 4] = __builtin_amdgcn_sudot4(false, q3, true, (int)v3[d], sumi[s + 4], false);
            sumi[s + 6] = __builtin_amdgcn_sudot4(false, q4, true, (int)v4[d], sumi[s + 6], false);
          }
        }
      } else {
        const unsigned scw[4] = {cur.x, cur.y, cur.z, cur.w};
        int isum = 0;
#pragma unroll
        for (int s = 0; s < 16; s++) {
          const int sc = (int)(int8_t)(scw[s >> 2] >> (8 * (s & 3)));
          isum += __mul24(sc, sumi[s] - 32 * xs[s]);
        }
        acc = __builtin_fmaf(sm.xd[sb[b]], f16_to_f32(d6[b]) * (float)isum, chain ? acc : 0.f);
        out[b] = acc;
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  template <int... K>
  __device__ __forceinline__ void steps(const Smem& sm, float (&out)[NB], std::integer_sequence<int, K...>) {
    int sumi[16];
    u4v hdr, qa, qb;
    float acc = 0.f;
    (step<K>(sm, out, sumi, hdr, qa, qb, acc), ...);
  }
  __device__ __forceinline__ void run(const Smem& sm, float (&out)[NB]) {
    steps(sm, out, std::make_integer_sequence<int, N>{});
  }
};

template <int NB, int D, unsigned Q6, unsigned CH>
__device__ __forceinline__ void stream_set(Stream<NB, D, Q6, CH>& st, int b, const QMat& m, int rb, int sb) {
  const unsigned blk = (unsigned)rb * m.SB + sb;
  const int lane = threadIdx.x & 31;
  st.base[b] = m.w + (blk * ((Q6 >> b) & 1 ? kQ6Chunks : kQ4Chunks) * 32 + lane);
  st.sb[b] = sb;
  if ((Q6 >> b) & 1) st.dptr[b] = m.d + blk * 32 + lane;
}

// Partials of NR row blocks x NM matrices meet in LDS, summed in wave order; threads t < NR * 32 get
// the total of row block t / 32, lane t % 32.
template <int NR, int NM>
__device__ __forceinline__ bool reduce_partials(Smem& sm, const float (&p)[NR][NM], float (&tot)[NM], int& r) {
  static_assert(NR * NM <= 12 && NR <= kWaves, "Smem.part, and one wave per row block's total");
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  lds_sync();
#pragma unroll
  for (int i = 0; i < NR; i++)
#pragma unroll
    for (int m = 0; m < NM; m++) sm.part[(wv * NR + i) * NM + m][lane] = p[i][m];
  lds_sync();
  if (t >= NR * 32) return false;
  r = t >> 5;
#pragma unroll
  for (int m = 0; m < NM; m++) {
    tot[m] = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; w++) tot[m] += sm.part[(w * NR + r) * NM + m][lane];
  }
  return true;
}

// --------------------------------------------------------------- exchange
// The partial is already in xc->out (system-scope stores, drained by the barrier). Block 0 raises
// the flag and waits for the host's result; every block can then read xc->in.
constexpr unsigned long long kExchangeTimeout = 500000000ull;  // 5 s of wall clock (100 MHz)

__device__ __forceinline__ void publish(const Args& a, int i, float v) {
  __hip_atomic_store(&a.xc->out[i], v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float received(const Args& a, int i) {
  return __hip_atomic_load(&a.xc->in[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ void exchange(const Args& a, unsigned& my_gen, unsigned& xseq, unsigned op, unsigned n) {
  grid_barrier(a, my_gen);
  xseq++;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __hip_atomic_store(&a.xc->op, op, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&a.xc->n, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    asm volatile(MK_WAIT_STORES_LDS ::: "memory");
    __hip_atomic_store(&a.xc->out_seq, xseq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const unsigned long long t0 = wall_clock64();
    while (__hip_atomic_load(&a.xc->in_seq, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != xseq)
      if (wall_clock64() - t0 > kExchangeTimeout) {
        __hip_atomic_store(a.bar_count + 1, 1u, __ATOMIC_RELAXED, AGENT);
        break;
      }
  }
  // the other blocks wait here while block 0 waits for the peers: give them block 0's budget
  grid_barrier(a, my_gen, kExchangeTimeout + kBarrierTimeout);
}

// --------------------------------------------------------------- attention
__device__ __forceinline__ unsigned kv_off(int pos, int g, int stride) { return ((unsigned)pos * stride + g) * kDim; }

__device__ __forceinline__ void head_norm_rope(const Smem& sm, float (&x)[4], const float* w) {
  const int lane = threadIdx.x & 31;
  if (!w) return;
  float ss = x[0] * x[0];
#pragma unroll
  for (int j = 1; j < 4; j++) ss = __builtin_fmaf(x[j], x[j], ss);
  const float inv = 1.f / sqrtf(wave_sum(ss) / (float)kDim + kEps);
  float y[4];
#pragma unroll
  for (int j = 0; j < 4; j++) y[j] = x[j] * inv * w[lane + 32 * j];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int i = lane + 32 * j;
    x[j] = __builtin_fmaf(y[j], sm.cs[i], -(y[j + 2] * sm.sn[i]));
    x[j + 2] = __builtin_fmaf(y[j + 2], sm.cs[i], y[j] * sm.sn[i]);
  }
}

// Local group j of shard r: its KV slot, and whether it is the slot's first group (which writes the new K, V).
__host__ __device__ inline int grp_slot(int r, int j) { return (grp_first(r) + j) / kIG - kv_first(r); }
__host__ __device__ inline bool grp_writes_kv(int r, int j) { return j == 0 || grp_slot(r, j) != grp_slot(r, j - 1); }

// One work item = (set of kSet local groups of 8 query heads, context chunk c) with attn_seq(ntok) of the set's
// groups, one after the other: from kSeqTokens on, all of them (the groups of a KV head: the later passes find
// the chunk's K and V in cache), below it one group per item (more items, shorter ones). Wave w = head 8 ig + w
// of group ig, on KV slot grp_slot(role, ig). The slot's first group writes the new token's K and V into the
// cache; every group takes them from LDS. Groups past this shard's count (zero weights) are skipped.
__device__ void attention(const Args& a, Smem& sm, const Layer& L, int layer, int pos, int nchunk, int chunk) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5, i = t & 127;
  const int sub = t >> 4, part = t & 15;
  const int ntok = pos + 1;
  const float scale = 1.f / sqrtf((float)kDim);

  if (t < kRot) {
    const float theta_scale = powf(kTheta, -2.f / (float)kDim);
    float theta = (float)pos;
    for (int k = 0; k < t; k++) theta *= theta_scale;
    sm.cs[t] = cosf(theta);
    sm.sn[t] = sinf(theta);
  }
  lds_sync();

  const int role = layer_role(a.rank, layer), st = kv_stride(role);
  const int seq = attn_seq(ntok), ngrp = grp_count(role);
  for (int it = blockIdx.x * seq; it < kGroups * nchunk; it += gridDim.x * seq) {
   for (int hg = 0; hg < seq; hg++) {
    const int idx = it + hg, c = idx / kSet % nchunk, ig = idx / kSet / nchunk * kSet + idx % kSet;
    if (ig >= ngrp) continue;
    const int g = grp_slot(role, ig);
    const int t0 = c * chunk, t1 = min(t0 + chunk, ntok);
    const bool last = t1 == ntok;

    float xq[4], xn[4];
#pragma unroll
    for (int j = 0; j < 4; j++) xq[j] = a.q[(8 * ig + wv) * kDim + lane + 32 * j];
    if (last && wv < 2) {
      const float* src = wv ? a.v : a.k;
#pragma unroll
      for (int j = 0; j < 4; j++) xn[j] = src[g * kDim + lane + 32 * j];
    }
    const uint16_t* kc = a.kcache[layer];
    const uint16_t* vc = a.vcache[layer];
    const int tok0 = t0 + sub;
    const bool pre = tok0 < t1 && tok0 != pos;
    u4v kk0 = {0, 0, 0, 0}, vv0 = {0, 0, 0, 0};
    if (pre) {
      kk0 = *(const u4v*)(kc + kv_off(tok0, g, st) + 8 * part);
      vv0 = *(const u4v*)(vc + kv_off(tok0, g, st) + 8 * part);
    }

    head_norm_rope(sm, xq, L.q_norm);
#pragma unroll
    for (int j = 0; j < 4; j++) sm.q[wv][lane + 32 * j] = xq[j];
    if (last && wv < 2) {
      head_norm_rope(sm, xn, wv ? nullptr : L.k_norm);
      uint16_t* dst = (wv ? a.vcache : a.kcache)[layer] + kv_off(pos, g, st);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint16_t h = f32_to_f16(xn[j]);
        if (grp_writes_kv(role, ig)) dst[lane + 32 * j] = h;
        sm.kvnew[wv][lane + 32 * j] = h;
      }
    }
    lds_sync();

    float q[8][8];
#pragma unroll
    for (int h = 0; h < 8; h++)
#pragma unroll
      for (int e = 0; e < 8; e++) q[h][e] = sm.q[h][8 * part + e];
    u4v kcur = tok0 == pos ? *(const u4v*)&sm.kvnew[0][8 * part] : kk0;
#pragma unroll 1
    for (int base = t0; base < t1; base += 16) {
      const int tok = base + sub, tn = tok + 16;
      u4v knext = {0, 0, 0, 0};
      if (tn < t1)
        knext = tn == pos ? *(const u4v*)&sm.kvnew[0][8 * part] : *(const u4v*)(kc + kv_off(tn, g, st) + 8 * part);
      float sc[8];
#pragma unroll
      for (int h = 0; h < 8; h++) sc[h] = 0.f;
      if (tok < t1) {
        const u4v kk = kcur;
        const unsigned kw[4] = {kk.x, kk.y, kk.z, kk.w};
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float kf = (e & 1) ? half_hi(kw[e >> 1]) : half_lo(kw[e >> 1]);
#pragma unroll
          for (int h = 0; h < 8; h++) sc[h] = __builtin_fmaf(q[h][e], kf, sc[h]);
        }
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1)
#pragma unroll
        for (int h = 0; h < 8; h++) sc[h] += __shfl_xor(sc[h], off);
      if (tok < t1 && part < 8) {
        float mine = sc[0];
#pragma unroll
        for (int h = 1; h < 8; h++)
          if (part == h) mine = sc[h];
        sm.p[part][tok - t0] = mine * scale;
      }
      kcur = knext;
    }
    lds_sync();

    const int n = t1 - t0;
    float mx = -INFINITY;
    for (int j = lane; j < n; j += 32) mx = fmaxf(mx, sm.p[wv][j]);
    mx = wave_max(mx);
    float ls = 0.f;
    for (int j = lane; j < n; j += 32) {
      const float e = expf(sm.p[wv][j] - mx);
      sm.p[wv][j] = e;
      ls += e;
    }
    ls = wave_sum(ls);
    if (lane == 0) {
      a.part_m[(8 * ig + wv) * kMaxSplits + c] = mx;
      a.part_l[(8 * ig + wv) * kMaxSplits + c] = ls;
    }
    lds_sync();

    float acc[8][8];
#pragma unroll
    for (int h = 0; h < 8; h++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[h][e] = 0.f;
    u4v vcur = tok0 == pos ? *(const u4v*)&sm.kvnew[1][8 * part] : vv0;
#pragma unroll 1
    for (int base = t0; base < t1; base += 16) {
      const int tok = base + sub, tn = tok + 16;
      u4v vnext = {0, 0, 0, 0};
      if (tn < t1)
        vnext = tn == pos ? *(const u4v*)&sm.kvnew[1][8 * part] : *(const u4v*)(vc + kv_off(tn, g, st) + 8 * part);
      if (tok < t1) {
        const u4v vv = vcur;
        const unsigned vw[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int h = 0; h < 8; h++) {
          const float pr = sm.p[h][tok - t0];
#pragma unroll
          for (int e = 0; e < 8; e++) {
            const float vf = (e & 1) ? half_hi(vw[e >> 1]) : half_lo(vw[e >> 1]);
            acc[h][e] = __builtin_fmaf(pr, vf, acc[h][e]);
          }
        }
      }
      vcur = vnext;
    }
    lds_sync();
    constexpr int kHalf = kWaves / 2;
#pragma unroll
    for (int h = 0; h < 8; h++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        acc[h][e] += __shfl_xor(acc[h][e], 16);
        if (lane < 16 && wv >= kHalf) sm.vred[wv - kHalf][h][8 * part + e] = acc[h][e];
      }
    lds_sync();
    if (lane < 16 && wv < kHalf)
#pragma unroll
      for (int h = 0; h < 8; h++)
#pragma unroll
        for (int e = 0; e < 8; e++) sm.vred[wv][h][8 * part + e] += acc[h][e];
    lds_sync();
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int hl = (t >> 7) + 2 * k, head = 8 * ig + hl;
      float out = 0.f;
#pragma unroll
      for (int w = 0; w < kHalf; w++) out += sm.vred[w][hl][i];
      a.part_acc[((unsigned)head * kMaxSplits + c) * kDim + i] = out;
    }

    asm volatile(MK_WAIT_STORES_LDS ::: "memory");
    __builtin_amdgcn_s_barrier();
    if (t == 0) sm.flag = __hip_atomic_fetch_add(&a.attn_cnt[ig], 1u, __ATOMIC_RELAXED, AGENT) == (unsigned)nchunk - 1;
    lds_sync();
    if (sm.flag) {
      asm volatile(MK_INV_DEV ::: "memory");
      const int head = 8 * ig + wv;
      const float* pm = a.part_m + head * kMaxSplits;
      const float* pl = a.part_l + head * kMaxSplits;
      const float m0 = lane < nchunk ? pm[lane] : -INFINITY, m1 = lane + 32 < nchunk ? pm[lane + 32] : -INFINITY;
      const float l0 = lane < nchunk ? pl[lane] : 0.f, l1 = lane + 32 < nchunk ? pl[lane + 32] : 0.f;
      const float M = wave_max(fmaxf(m0, m1));
      const float s0 = expf(m0 - M), s1 = expf(m1 - M);
      float l = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int cc = 0; cc < nchunk; cc++) {
        const float sc = __shfl(cc < 32 ? s0 : s1, cc & 31), lc = __shfl(cc < 32 ? l0 : l1, cc & 31);
        l = __builtin_fmaf(lc, sc, l);
        const float4 x = *(const float4*)(a.part_acc + ((unsigned)head * kMaxSplits + cc) * kDim + 4 * lane);
        o[0] = __builtin_fmaf(x.x, sc, o[0]);
        o[1] = __builtin_fmaf(x.y, sc, o[1]);
        o[2] = __builtin_fmaf(x.z, sc, o[2]);
        o[3] = __builtin_fmaf(x.w, sc, o[3]);
      }
      *(float4*)(a.attn + head * kDim + 4 * lane) = make_float4(o[0] / l, o[1] / l, o[2] / l, o[3] / l);
      if (t == 0) __hip_atomic_store(&a.attn_cnt[ig], 0u, __ATOMIC_RELAXED, AGENT);
    }
    lds_sync();
   }
  }
}

// Items are numbered (set * nchunk + c) * kSet + group, attn_seq(ntok) consecutive ones per block pass.
__host__ __device__ inline void shard_attn_split(int ntok, int& nchunk, int& chunk) {
  ::attn_split(ntok, ntok >= kSeqTokens ? kGroups / kSet : kGroups, kGrid, kMaxChunk, nchunk, chunk);
}

// ------------------------------------------------------------------- router
// logit_e = inv_rms * sum_i R[e][i] w[i] x[i]: block partials over its rows of x into router_part[e][hb]
// and sum x^2 into router_ss[hb] (with sm.xw = x * ffn_norm of the block's rows already in LDS).
__device__ void router_partials(const Args& a, const Smem& sm, const Layer& L) {
  const int t = threadIdx.x, re = t >> 1, rh = t & 1;
#pragma unroll
  for (int j = 0; j < kOR; j++) {
    const int hb = blockIdx.x + kGrid * j;
    const float4* R = (const float4*)(L.router + re * kHidden + 32 * hb + 16 * rh);
    const float4 r0 = R[0], r1 = R[1], r2 = R[2], r3 = R[3];
    const float rr[16] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w,
                          r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w};
    const float* xw = sm.xw + 32 * j + 16 * rh;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) acc = __builtin_fmaf(rr[k], xw[k], acc);
    acc += __shfl_xor(acc, 1);
    if (rh == 0) a.router_part[re * kHB + hb] = acc;
  }
}

// Threads t < kOR * 32: x rows of row block hb = b + kGrid * (t / 32) become xn; their x * ffn_norm go to
// LDS and their sum of squares to router_ss.
__device__ __forceinline__ void set_rows(const Args& a, Smem& sm, const Layer& L, int r, float xn, bool router) {
  const int lane = threadIdx.x & 31, hb = blockIdx.x + kGrid * r;
  a.x[hb * 32 + lane] = xn;
  if (router) {
    sm.xw[r * 32 + lane] = xn * L.ffn_norm[hb * 32 + lane];
    const float ss = wave_sum(xn * xn);
    if (lane == 0) a.router_ss[hb] = ss;
  }
}

// Sums the kHB partials per expert, top 8 by rank (ties to the lower index), softmax over all 128
// renormalized over the 8, in rank order. Every block does the same.
__device__ void route(const Args& a, Smem& sm) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  float tot = 0.f;
  if (t < kExperts) {
    const float4* P = (const float4*)(a.router_part + t * kHB);
#pragma unroll 8
    for (int j = 0; j < kHB / 4; j++) {
      const float4 pp = P[j];
      tot = ((tot + pp.x) + pp.y + pp.z) + pp.w;
    }
  } else if (wv == kExperts / 32) {
    float s = a.router_ss[lane];
#pragma unroll
    for (int j = 1; j < kHB / 32; j++) s += a.router_ss[lane + 32 * j];
    const float ss = wave_sum(s);
    if (lane == 0) sm.red[16] = 1.f / sqrtf(ss / (float)kHidden + kEps);
  }
  lds_sync();
  const float lg = tot * sm.red[16];
  if (t < kExperts) {
    const unsigned u = __float_as_uint(lg), ord = u ^ ((unsigned)((int)u >> 31) | 0x80000000u);
    sm.rk[t] = ((unsigned long long)ord << 8) | (unsigned)(kExperts - 1 - t);
    const float m = wave_max(lg);
    if (lane == 0) sm.red[wv] = m;
  }
  lds_sync();
  float ex = 0.f;
  if (t < kExperts) {
    const float M = fmaxf(fmaxf(sm.red[0], sm.red[1]), fmaxf(sm.red[2], sm.red[3]));
    ex = expf(lg - M);
    const float s = wave_sum(ex);
    if (lane == 0) sm.red[8 + wv] = s;
  }
  lds_sync();
  if (t < kExperts) {
    const float S = ((sm.red[8] + sm.red[9]) + sm.red[10]) + sm.red[11];
    int rank = 0;
    const unsigned long long me = sm.rk[t];
#pragma unroll 1
    for (int j0 = 0; j0 < kExperts; j0 += 32) {
      ulonglong2 o[16];
#pragma unroll
      for (int k = 0; k < 16; k++) o[k] = *(const ulonglong2*)&sm.rk[j0 + 2 * k];
#pragma unroll
      for (int k = 0; k < 16; k++) rank += (o[k].x > me) + (o[k].y > me);
    }
    if (rank < kTopK) {
      sm.sel[rank] = t;
      sm.pk[rank] = ex / S;
    }
  }
  lds_sync();
  if (t < kTopK) {
    float psum = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++) psum += sm.pk[k];
    sm.selw[t] = sm.pk[t] / psum;
  }
  lds_sync();
}

// ------------------------------------------------------------------ stages
// Element e of Q4_K row m in the GGUF layout (the embedding).
__device__ float q4_value_raw(const uint8_t* W, int m, int e) {
  const uint8_t* b = W + ((size_t)m * kHSB + (e >> 8)) * kQ4Bytes;
  const int kk = e & 255, j = kk >> 6, hi = (kk >> 5) & 1, l = kk & 31;
  const int s = 2 * j + hi;
  const uint8_t* sc = b + 4;
  int scv, mn;
  if (s < 4) {
    scv = sc[s] & 63;
    mn = sc[s + 4] & 63;
  } else {
    scv = (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4);
    mn = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
  }
  const float d = f16_to_f32((uint16_t)(b[0] | b[1] << 8)), dmin = f16_to_f32((uint16_t)(b[2] | b[3] << 8));
  const int q = hi ? b[16 + 32 * j + l] >> 4 : b[16 + 32 * j + l] & 15;
  return __builtin_fmaf(d * (float)scv, (float)q, -(dmin * (float)mn));
}

// qkv: V row block vrb when VK is 1 (Q4_K) or 2 (Q6_K), then Q and K row blocks it = b + kGrid j (NI of
// them for this block), all in one stream; waves take superblocks w, w + 8, ... of each. V goes first:
// a Q6_K block at the end of a long stream makes the compiler spill.
template <int NI, int VK>
__device__ __forceinline__ void qkv_items(const Args& a, Smem& sm, const Layer& L, float (&v)[kSPW][8],
                                          const float (&w)[kSPW][8], int start, int vrb) {
  constexpr int NV = VK ? 1 : 0, NR = NI + NV;
  constexpr unsigned Q6 = VK == 2 ? (1u << kSPW) - 1 : 0u;
  const int b0 = blockIdx.x, wv = threadIdx.x >> 5, lane = threadIdx.x & 31;
  Stream<NR * kSPW, kD, Q6, chain_mask(NR, kSPW)> st;
  if constexpr (VK != 0) {
    const QMat m = {L.v, L.v_d, kHSB};
#pragma unroll
    for (int i = 0; i < kSPW; i++) stream_set(st, i, m, vrb, wv + kWaves * i);
  }
#pragma unroll
  for (int j = 0; j < NI; j++) {
    const int it = start + j;
    const bool isq = it < kQS / 32;
    const QMat m = {isq ? L.q : L.k, nullptr, kHSB};
#pragma unroll
    for (int i = 0; i < kSPW; i++) stream_set(st, (NV + j) * kSPW + i, m, isq ? it : it - kQS / 32, wv + kWaves * i);
  }
  st.pre();
  groups_rmsnorm(sm, v, w);
  groups_quantize(sm, v);
  float o[NR * kSPW];
  st.run(sm, o);
  float p[NR][1];
#pragma unroll
  for (int j = 0; j < NR; j++) p[j][0] = o[j * kSPW + kSPW - 1];
  float tot[1];
  int r;
  if (reduce_partials(sm, p, tot, r)) {
    if (r < NV) {
      a.v[vrb * 32 + lane] = tot[0];
    } else {
      const int it = start + r - NV;
      if (it < kQS / 32) a.q[it * 32 + lane] = tot[0];
      else a.k[(it - kQS / 32) * 32 + lane] = tot[0];
    }
  }
}

template <int NI>
__device__ __forceinline__ void qkv_stage(const Args& a, Smem& sm, const Layer& L, float (&v)[kSPW][8],
                                          const float (&w)[kSPW][8], int start, int vrb) {
  if (vrb < 0) qkv_items<NI, 0>(a, sm, L, v, w, start, vrb);
  else if (L.v_d) qkv_items<NI, 2>(a, sm, L, v, w, start, vrb);
  else qkv_items<NI, 1>(a, sm, L, v, w, start, vrb);
}

template <bool XCHG>
__device__ __forceinline__ void layer_attention(const Args& a, Smem& sm, const Layer& L, int layer,
                                               int pos, int nchunk, int chunk, unsigned& my_gen,
                                               unsigned long long* stamps, int& ns) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5, b0 = blockIdx.x;
  {  // qkv
    float v[kSPW][8], w[kSPW][8];
    load_groups(a.x, v);
    load_groups(L.attn_norm, w);
    if (b0 < kQBlocks) {
      const int start = b0 * kQLo + min(b0, kQExtra);
      if (b0 < kQExtra) qkv_stage<kQLo + 1>(a, sm, L, v, w, start, -1);
      else qkv_stage<kQLo>(a, sm, L, v, w, start, -1);
    } else {
      qkv_stage<kVQ>(a, sm, L, v, w, kQRest + (b0 - kQBlocks) * kVQ, b0 - kQBlocks);
    }
  }
  grid_barrier(a, my_gen);
  stamp(stamps, ns);

  attention(a, sm, L, layer, pos, nchunk, chunk);
  grid_barrier(a, my_gen);
  stamp(stamps, ns);

  {  // o: row blocks b + kGrid j; wave set ws = w / kOWR takes j in [ws kORW, ..), its wave i = w % kOWR
     // superblocks i, i + kOWR, ... of this shard's columns
    float v[kOQG][8];
    load_groups(a.attn, v);
    const QMat mo = {L.o, nullptr, kQSB};
    const int ws = kORS == 1 ? 0 : wv / kOWR, wi = kORS == 1 ? wv : wv % kOWR;
    Stream<kORW * kOSW, kD, 0u, chain_mask(kORW, kOSW)> st;
#pragma unroll
    for (int j = 0; j < kORW; j++)
#pragma unroll
      for (int i = 0; i < kOSW; i++)
        stream_set(st, j * kOSW + i, mo, b0 + kGrid * (ws * kORW + j), wi + kOWR * i);
    st.pre();
    groups_quantize(sm, v);
    float o[kORW * kOSW];
    st.run(sm, o);
    float p[kOR][1];
#pragma unroll
    for (int j = 0; j < kOR; j++) p[j][0] = j / kORW == ws ? o[j % kORW * kOSW + kOSW - 1] : 0.f;
    float tot[1];
    int r;
    if (reduce_partials(sm, p, tot, r)) {
      const int rb = b0 + kGrid * r;
      if constexpr (XCHG) publish(a, rb * 32 + lane, tot[0]);
      else set_rows(a, sm, L, r, a.x[rb * 32 + lane] + tot[0], true);
    }
    if constexpr (!XCHG) {
      lds_sync();
      router_partials(a, sm, L);
    }
  }
}

template <bool XCHG>
__device__ __forceinline__ void layer_moe(const Args& a, Smem& sm, const Layer& L, int layer,
                                         unsigned& my_gen, unsigned long long* stamps, int& ns) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5, b0 = blockIdx.x;
  {  // gate, up: pairs p = b0 + kGrid j (slot p / kRBE, row block p % kRBE of that expert's shard rows)
    float v[kSPW][8], w[kSPW][8];
    load_groups(a.x, v);
    load_groups(L.ffn_norm, w);
    route(a, sm);
    if (a.trace_sel && b0 == 0 && t < kTopK) a.trace_sel[layer * kTopK + t] = sm.sel[t];
#pragma unroll
    for (int h = 0; h < kPPB / kPPC; h++) {
      Stream<kPPC * 2 * kSPW, kD, 0u, chain_mask(kPPC * 2, kSPW)> st;
#pragma unroll
      for (int j = 0; j < kPPC; j++) {
        const int pr = b0 + kGrid * (kPPC * h + j), rb = sm.sel[pr / kRBE] * kRBE + pr % kRBE;
#pragma unroll
        for (int i = 0; i < kSPW; i++) {
          stream_set(st, (2 * j) * kSPW + i, QMat{L.gate, nullptr, kHSB}, rb, wv + kWaves * i);
          stream_set(st, (2 * j + 1) * kSPW + i, QMat{L.up, nullptr, kHSB}, rb, wv + kWaves * i);
        }
      }
      st.pre();
      if (h == 0) {
        groups_rmsnorm(sm, v, w);
        groups_quantize(sm, v);
      }
      float o[kPPC * 2 * kSPW];
      st.run(sm, o);
      float p[kPPC][2];
#pragma unroll
      for (int j = 0; j < kPPC; j++) {
        p[j][0] = o[(2 * j) * kSPW + kSPW - 1];
        p[j][1] = o[(2 * j + 1) * kSPW + kSPW - 1];
      }
      float tot[2];
      int r;
      if (reduce_partials(sm, p, tot, r)) {
        const int pr = b0 + kGrid * (kPPC * h + r);
        const float g = tot[0], u = tot[1];
        a.act[(pr / kRBE) * kFFP + a.act_off + (pr % kRBE) * 32 + lane] = (g / (1.f + expf(-g))) * u;
      }
    }
  }
  grid_barrier(a, my_gen);
  stamp(stamps, ns);

  {  // down: wave w = slot w, row blocks hb = b0 + kGrid h of expert sel[w] over this shard's kDSB superblocks,
     // kDH of them per stream; the routing-weighted sum in slot order
#pragma unroll
    for (int pass = 0; pass < kOR / kDH; pass++) {
      auto body = [&](auto& st) {
#pragma unroll
        for (int j = 0; j < kDH; j++)
#pragma unroll
          for (int i = 0; i < kDSB; i++) {
            const int hb = b0 + kGrid * (kDH * pass + j);
            stream_set(st, j * kDSB + i, QMat{L.down, L.down_d, kDSB}, sm.sel[wv] * kHB + hb, i);
            st.sb[j * kDSB + i] = kDSB * wv + i;
          }
        st.pre();
        if (pass == 0) {
          float v[kDSB][8];
#pragma unroll
          for (int i = 0; i < kDSB; i++) {
            const float4* src = (const float4*)(a.act + wv * kFFP + 256 * i + 8 * lane);
            const float4 x0 = src[0], x1 = src[1];
            v[i][0] = x0.x, v[i][1] = x0.y, v[i][2] = x0.z, v[i][3] = x0.w;
            v[i][4] = x1.x, v[i][5] = x1.y, v[i][6] = x1.z, v[i][7] = x1.w;
          }
#pragma unroll
          for (int i = 0; i < kDSB; i++) quantize_group(sm, kDSB * wv + i, v[i]);
          lds_sync();
        }
        float o[kDH * kDSB];
        st.run(sm, o);
#pragma unroll
        for (int j = 0; j < kDH; j++) sm.part[(kDH * pass + j) * kWaves + wv][lane] = o[j * kDSB + kDSB - 1];
      };
      if (L.down_q6) {
        Stream<kDH * kDSB, kD, (1u << (kDH * kDSB)) - 1, chain_mask(kDH, kDSB)> st;
        body(st);
      } else {
        Stream<kDH * kDSB, kD, 0u, chain_mask(kDH, kDSB)> st;
        body(st);
      }
    }
    lds_sync();
    if (t < kOR * 32) {
      const int h = t >> 5, hb = b0 + kGrid * h;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < kTopK; k++) acc = __builtin_fmaf(sm.selw[k], sm.part[h * kWaves + k][lane], acc);
      if constexpr (XCHG) publish(a, hb * 32 + lane, acc);
      else a.x[hb * 32 + lane] += acc;
    }
  }
}

template <bool XCHG>
__device__ __forceinline__ void output_logits(const Args& a, Smem& sm, float* logits, unsigned& my_gen,
                                             unsigned long long* stamps, int& ns) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5, b0 = blockIdx.x;
  const int wave = wave_index(), nw = wave_count();
  {  // LM head over this shard's vocabulary rows + argmax
    const QMat lm = {a.lm, a.lm_d, kHSB};
    float v[kSPW][8], w[kSPW][8];
    load_groups(a.x, v);
    load_groups(a.out_norm, w);
    B6 w0;
    if (wave < a.vocab / 32) fetch_q6(w0, lm.w, lm.d, wave, lm.SB, 0);
    groups_rmsnorm(sm, v, w);
    groups_quantize(sm, v);
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int task = wave; task < a.vocab / 32; task += nw) {
      const bool first = task == wave;
      const float y = lm_row_dot(lm, task, sm, first ? 1 : 0, first ? dot_q6(w0, 0, sm, 0.f) : 0.f);
      const int m = task * 32 + lane;
      logits[m] = y;
      if (y > best || (y == best && m < bidx)) {
        best = y;
        bidx = m;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ob = __shfl_xor(best, off);
      const int oi = __shfl_xor(bidx, off);
      if (ob > best || (ob == best && oi < bidx)) {
        best = ob;
        bidx = oi;
      }
    }
    lds_sync();
    if (lane == 0) {
      sm.red[wv] = best;
      sm.best_i[wv] = bidx;
    }
    lds_sync();
    if (t == 0) {
      for (int k = 1; k < kWaves; k++)
        if (sm.red[k] > best || (sm.red[k] == best && sm.best_i[k] < bidx)) {
          best = sm.red[k];
          bidx = sm.best_i[k];
        }
      a.best_v[b0] = best;
      a.best_i[b0] = bidx;
    }
  }
  grid_barrier(a, my_gen);
  stamp(stamps, ns);
  if (t == 0) {
    float best = a.best_v[0];
    int bidx = a.best_i[0];
    for (int b = 1; b < (int)gridDim.x; b++)
      if (a.best_v[b] > best || (a.best_v[b] == best && a.best_i[b] < bidx)) {
        best = a.best_v[b];
        bidx = a.best_i[b];
      }
    sm.best_i[0] = bidx + a.vocab_off;
    if (XCHG && b0 == 0) {
      publish(a, 0, best);
      publish(a, 1, __int_as_float(bidx + a.vocab_off));
    }
  }
}

template <bool XCHG>
__device__ __forceinline__ int decode_step(const Args& a, Smem& sm, int pos, int token, unsigned& my_gen,
                                           unsigned& xseq, unsigned long long* stamps, float* trace, float* logits) {
  const int t = threadIdx.x, lane = t & 31;
  const int b0 = blockIdx.x;
  int ns = 0;
  stamp(stamps, ns);

  for (int e = b0 * kBlock + t; e < kHidden; e += gridDim.x * kBlock) a.x[e] = q4_value_raw(a.emb, token, e);
  grid_barrier(a, my_gen);
  stamp(stamps, ns);

  int nchunk, chunk;
  shard_attn_split(pos + 1, nchunk, chunk);

  for (int layer = 0; layer < a.n_layers; layer++) {
    const Layer L = a.layers[layer];
    if (trace && b0 == 0)
      for (int e = t; e < kHidden; e += kBlock) trace[(size_t)layer * kHidden + e] = a.x[e];

    layer_attention<XCHG>(a, sm, L, layer, pos, nchunk, chunk, my_gen, stamps, ns);
    if constexpr (XCHG) {
      exchange(a, my_gen, xseq, kOpSum, kHidden);
      stamp(stamps, ns);
      if (t < kOR * 32) {
        const int r = t >> 5, rb = b0 + kGrid * r;
        set_rows(a, sm, L, r, a.x[rb * 32 + lane] + received(a, rb * 32 + lane), true);
      }
      lds_sync();
      router_partials(a, sm, L);
    }
    grid_barrier(a, my_gen);
    stamp(stamps, ns);

    layer_moe<XCHG>(a, sm, L, layer, my_gen, stamps, ns);
    if constexpr (XCHG) {
      exchange(a, my_gen, xseq, kOpSum, kHidden);
      stamp(stamps, ns);
      if (t < kOR * 32) {
        const int r = t >> 5, rb = b0 + kGrid * r;
        set_rows(a, sm, L, r, a.x[rb * 32 + lane] + received(a, rb * 32 + lane), false);
      }
    }
    grid_barrier(a, my_gen);
    stamp(stamps, ns);
  }

  if (trace && b0 == 0)
    for (int e = t; e < kHidden; e += kBlock) trace[(size_t)a.n_layers * kHidden + e] = a.x[e];

  output_logits<XCHG>(a, sm, logits, my_gen, stamps, ns);
  if constexpr (XCHG) {
    exchange(a, my_gen, xseq, kOpArgmax, 2);
    stamp(stamps, ns);
    if (t == 0) sm.best_i[0] = __float_as_int(received(a, 1));
  }
  lds_sync();
  return sm.best_i[0];
}

__global__ void __launch_bounds__(kBlock) decode(Args a) {
  __shared__ Smem sm;
  unsigned my_gen = a.gen_base, xseq = a.xseq_base;
  int token = *a.token;
  for (int step = 0; step < a.n_steps; step++) {
    if (a.inputs) token = a.inputs[step];
    float* logits = a.logits_all ? a.logits_all + (size_t)step * a.vocab : a.logits;
    unsigned long long* st = step == 0 ? a.stamps : nullptr;
    float* tr = step == 0 ? a.trace : nullptr;
    token = decode_step<(kShards > 1)>(a, sm, a.pos + step, token, my_gen, xseq, st, tr, logits);
    if (blockIdx.x == 0 && threadIdx.x == 0) a.tokens_out[step] = token;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *a.token = token;
  mk_status(a.status, a.bar_count);
}

#if MK_STAGE_SPLIT
// Each stage returns its partial before the host exchanges it. The next stage
// consumes the reduced result; no GPU thread waits for a host notification.
__global__ void __launch_bounds__(kBlock) decode_attention_stage(Args a, int layer) {
  __shared__ Smem sm;
  unsigned my_gen = a.gen_base;
  const int t = threadIdx.x, lane = t & 31, b0 = blockIdx.x;
  const Layer L = a.layers[layer];
  if (layer == 0) {
    const int token = a.inputs ? a.inputs[0] : *a.token;
    for (int e = b0 * kBlock + t; e < kHidden; e += gridDim.x * kBlock)
      a.x[e] = q4_value_raw(a.emb, token, e);
  } else if (t < kOR * 32) {
    const int r = t >> 5, rb = b0 + kGrid * r;
    set_rows(a, sm, L, r, a.x[rb * 32 + lane] + received(a, rb * 32 + lane), false);
  }
  grid_barrier(a, my_gen);
  if (a.trace && b0 == 0)
    for (int e = t; e < kHidden; e += kBlock) a.trace[(size_t)layer * kHidden + e] = a.x[e];
  int nchunk, chunk, ns = 0;
  shard_attn_split(a.pos + 1, nchunk, chunk);
  layer_attention<true>(a, sm, L, layer, a.pos, nchunk, chunk, my_gen, nullptr, ns);
}

__global__ void __launch_bounds__(kBlock) decode_moe_stage(Args a, int layer) {
  __shared__ Smem sm;
  unsigned my_gen = a.gen_base;
  const int t = threadIdx.x, lane = t & 31, b0 = blockIdx.x;
  const Layer L = a.layers[layer];
  if (t < kOR * 32) {
    const int r = t >> 5, rb = b0 + kGrid * r;
    set_rows(a, sm, L, r, a.x[rb * 32 + lane] + received(a, rb * 32 + lane), true);
  }
  lds_sync();
  router_partials(a, sm, L);
  grid_barrier(a, my_gen);
  int ns = 0;
  layer_moe<true>(a, sm, L, layer, my_gen, nullptr, ns);
}

__global__ void __launch_bounds__(kBlock) decode_head_stage(Args a) {
  __shared__ Smem sm;
  unsigned my_gen = a.gen_base;
  const int t = threadIdx.x, lane = t & 31, b0 = blockIdx.x;
  const Layer L = a.layers[a.n_layers - 1];
  if (t < kOR * 32) {
    const int r = t >> 5, rb = b0 + kGrid * r;
    set_rows(a, sm, L, r, a.x[rb * 32 + lane] + received(a, rb * 32 + lane), false);
  }
  grid_barrier(a, my_gen);
  if (a.trace && b0 == 0)
    for (int e = t; e < kHidden; e += kBlock) a.trace[(size_t)a.n_layers * kHidden + e] = a.x[e];
  int ns = 0;
  output_logits<true>(a, sm, a.logits_all ? a.logits_all : a.logits, my_gen, nullptr, ns);
}
#endif

// Grid barriers and exchanges per decoded token.
static int barriers_per_step(int n_layers, bool xchg) { return xchg ? 9 * n_layers + 4 : 5 * n_layers + 2; }
static int exchanges_per_step(int n_layers) { return 2 * n_layers + 1; }

// -------------------------------------------------------------------- host
enum Kind { kQ4 = 0, kQ6 = 1 };
static size_t block_bytes(Kind k) { return k == kQ4 ? kQ4Bytes : kQ6Bytes; }

// A whole-model tensor in llama.cpp's layout (row-major blocks), not owned: the mapped GGUF or a
// random buffer.
struct TView {
  Kind kind;
  int M, K;
  const uint8_t* data;
  size_t row_bytes() const { return (size_t)(K / 256) * block_bytes(kind); }
};

struct Tensor {
  Kind kind;
  int M = 0, K = 0;
  std::vector<uint8_t> packed;
  u4v* dev = nullptr;
  uint16_t* dev_d = nullptr;
};
static size_t tensor_bytes(const Tensor& t) { return (size_t)t.M * (t.K / 256) * block_bytes(t.kind); }

static Tensor rows_of(const TView& v, int r0, int n) {
  Tensor t{v.kind, n, v.K};
  t.packed.assign(v.data + (size_t)r0 * v.row_bytes(), v.data + (size_t)(r0 + n) * v.row_bytes());
  return t;
}
// Superblocks sb0 .. sb0 + nsb of every row.
static Tensor cols_of(const TView& v, int sb0, int nsb) {
  const size_t bb = block_bytes(v.kind);
  Tensor t{v.kind, v.M, nsb * 256};
  t.packed.resize((size_t)v.M * nsb * bb);
  for (int m = 0; m < v.M; m++)
    memcpy(&t.packed[(size_t)m * nsb * bb], v.data + (size_t)m * v.row_bytes() + sb0 * bb, nsb * bb);
  return t;
}
// Zero blocks (d = dmin = 0: the values are 0) up to M rows, or up to nsb superblocks of every row.
static Tensor pad_rows(Tensor t, int M) {
  t.packed.resize((size_t)M * (t.K / 256) * block_bytes(t.kind), 0);
  t.M = M;
  return t;
}
static Tensor pad_cols(const Tensor& t, int nsb) {
  const size_t bb = block_bytes(t.kind), have = (size_t)(t.K / 256) * bb;
  Tensor r{t.kind, t.M, nsb * 256};
  r.packed.assign((size_t)t.M * nsb * bb, 0);
  for (int m = 0; m < t.M; m++) memcpy(&r.packed[(size_t)m * nsb * bb], &t.packed[(size_t)m * have], have);
  return r;
}
// Rows r0 .. r0 + n of each expert (F rows per expert).
static Tensor expert_rows(const TView& v, int F, int r0, int n) {
  const int E = v.M / F;
  Tensor t{v.kind, E * n, v.K};
  t.packed.resize((size_t)E * n * v.row_bytes());
  for (int e = 0; e < E; e++)
    memcpy(&t.packed[(size_t)e * n * v.row_bytes()], v.data + ((size_t)e * F + r0) * v.row_bytes(),
           (size_t)n * v.row_bytes());
  return t;
}

// Q4_K sub-block s covers elements 32 s ..; its 6-bit scale and min are packed with the others.
static void q4_scales(const uint8_t* q, int* sc, int* mn) {
  for (int j = 0; j < 8; j++) {
    if (j < 4) {
      sc[j] = q[j] & 63;
      mn[j] = q[j + 4] & 63;
    } else {
      sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
      mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
  }
}
static void q4_pack_scales(uint8_t* q, const int* sc, const int* mn) {
  for (int i = 0; i < 4; i++) {
    q[i] = (uint8_t)((sc[i] & 63) | ((sc[i + 4] >> 4) << 6));
    q[i + 4] = (uint8_t)((mn[i] & 63) | ((mn[i + 4] >> 4) << 6));
    q[i + 8] = (uint8_t)((sc[i + 4] & 15) | ((mn[i + 4] & 15) << 4));
  }
}

// Shard r's columns of every expert's down: whole superblocks covering expert rows
// r * kFFS .. (r + 1) * kFFS, scales of the sub-blocks outside them zero.
static Tensor down_slice(const TView& v, int r) {
  const int sb0 = down_first_sb(r), lo = r * kFFS, hi = lo + kFFS;
  Tensor t = cols_of(v, sb0, kDSB);
  const size_t bb = block_bytes(t.kind);
  for (int m = 0; m < t.M; m++)
    for (int i = 0; i < kDSB; i++) {
      uint8_t* b = &t.packed[((size_t)m * kDSB + i) * bb];
      const int k0 = (sb0 + i) * 256;
      if (t.kind == kQ4) {
        int sc[8], mn[8];
        q4_scales(b + 4, sc, mn);
        for (int s = 0; s < 8; s++)
          if (k0 + 32 * s < lo || k0 + 32 * s >= hi) sc[s] = mn[s] = 0;
        q4_pack_scales(b + 4, sc, mn);
      } else {
        for (int s = 0; s < 16; s++) {
          const int e = k0 + 128 * (s >> 3) + 32 * ((s & 7) >> 1) + 16 * (s & 1);
          if (e < lo || e >= hi) b[192 + s] = 0;
        }
      }
    }
  return t;
}

static void upload(Tensor& t, bool keep_host) {
  const int SB = t.K / 256, chunks = t.kind == kQ4 ? kQ4Chunks : kQ6Chunks, bb = (int)block_bytes(t.kind);
  std::vector<uint8_t> tiled((size_t)t.M * SB * chunks * 16);
  std::vector<uint16_t> d(t.kind == kQ6 ? (size_t)t.M * SB : 0);
  for (int m = 0; m < t.M; m++)
    for (int sb = 0; sb < SB; sb++) {
      const uint8_t* blk = t.packed.data() + ((size_t)m * SB + sb) * bb;
      const size_t tb = (size_t)(m / 32) * SB + sb;
      for (int c = 0; c < chunks; c++) memcpy(&tiled[((tb * chunks + c) * 32 + m % 32) * 16], blk + c * 16, 16);
      if (t.kind == kQ6) memcpy(&d[tb * 32 + m % 32], blk + 208, 2);
    }
  CHECK(hipMalloc(&t.dev, tiled.size()));
  CHECK(hipMemcpy(t.dev, tiled.data(), tiled.size(), hipMemcpyHostToDevice));
  if (t.kind == kQ6) {
    CHECK(hipMalloc(&t.dev_d, d.size() * 2));
    CHECK(hipMemcpy(t.dev_d, d.data(), d.size() * 2, hipMemcpyHostToDevice));
  }
  if (!keep_host) std::vector<uint8_t>().swap(t.packed);
}

// ------------------------------------------------------------ whole model
struct FullLayer {
  TView q, k, v, o, gate, up, down;
  const float *router, *attn_norm, *ffn_norm, *q_norm, *k_norm;
};
struct FullModel {
  int n_layers = 0, vocab = 0;
  std::vector<FullLayer> layers;
  TView emb, lm;
  const float* out_norm = nullptr;
  std::vector<std::vector<uint8_t>> owned;  // random models
  std::vector<std::unique_ptr<Gguf>> parts;  // GGUF models, mapped
};

// part-00001-of-0000N.gguf opens all N parts; any other name one file.
static void open_gguf_parts(FullModel& fm, const std::string& path) {
  const size_t at = path.find("-00001-of-");
  int n = 1;
  if (at != std::string::npos) n = atoi(path.c_str() + at + 10);
  for (int i = 1; i <= n; i++) {
    std::string p = path;
    if (n > 1) {
      char num[8];
      snprintf(num, sizeof(num), "%05d", i);
      p.replace(at + 1, 5, num);
    }
    fm.parts.emplace_back(new Gguf());
    read_gguf(*fm.parts.back(), p.c_str());
  }
}

static void model_from_gguf(FullModel& fm, const std::string& path, int max_layers) {
  open_gguf_parts(fm, path);
  const Gguf& g = *fm.parts[0];
  if (!g.strs.count("general.architecture") || g.strs.at("general.architecture") != "qwen3moe") {
    fprintf(stderr, "gguf: not a qwen3moe model\n");
    exit(1);
  }
  g.expect("qwen3moe.embedding_length", kHidden);
  g.expect("qwen3moe.attention.head_count", kHeads);
  g.expect("qwen3moe.attention.head_count_kv", kKV);
  g.expect("qwen3moe.attention.key_length", kDim);
  g.expect("qwen3moe.attention.value_length", kDim);
  g.expect("qwen3moe.rope.freq_base", kTheta);
  g.expect("qwen3moe.attention.layer_norm_rms_epsilon", kEps);
  g.expect("qwen3moe.expert_count", kExperts);
  g.expect("qwen3moe.expert_used_count", kTopK);
  g.expect("qwen3moe.expert_feed_forward_length", kFFE);
  auto find = [&](const std::string& name) -> std::pair<const Gguf*, const GgufTensor*> {
    for (auto& p : fm.parts)
      if (p->has(name)) return {p.get(), &p->tensor(name)};
    fprintf(stderr, "gguf: missing tensor %s\n", name.c_str());
    exit(1);
  };
  auto quant = [&](const std::string& name, int M, int K) {
    auto [p, t] = find(name);
    Kind kind = kQ4;
    if (t->type == kQ6_K) kind = kQ6;
    else if (t->type != kQ4_K) {
      fprintf(stderr, "gguf: %s has type %u, only Q4_K and Q6_K are supported\n", name.c_str(), t->type);
      exit(1);
    }
    size_t rows = 1;
    for (size_t i = 1; i < t->dims.size(); i++) rows *= t->dims[i];
    if ((int)t->dims[0] != K || (M > 0 && (int)rows != M)) {
      fprintf(stderr, "gguf: %s has an unexpected shape\n", name.c_str());
      exit(1);
    }
    return TView{kind, (int)rows, K, p->data(*t)};
  };
  auto f32 = [&](const std::string& name, size_t n) {
    auto [p, t] = find(name);
    size_t numel = 1;
    for (auto d : t->dims) numel *= d;
    if (t->type != kF32 || numel != n) {
      fprintf(stderr, "gguf: %s is not F32[%zu]\n", name.c_str(), n);
      exit(1);
    }
    return (const float*)p->data(*t);
  };
  fm.n_layers = std::min((int)g.num("qwen3moe.block_count"), max_layers);
  fm.layers.resize(fm.n_layers);
  for (int l = 0; l < fm.n_layers; l++) {
    FullLayer& L = fm.layers[l];
    const std::string b = "blk." + std::to_string(l) + ".";
    L.q = quant(b + "attn_q.weight", kHeads * kDim, kHidden);
    L.k = quant(b + "attn_k.weight", kKV * kDim, kHidden);
    L.v = quant(b + "attn_v.weight", kKV * kDim, kHidden);
    L.o = quant(b + "attn_output.weight", kHidden, kHeads * kDim);
    L.gate = quant(b + "ffn_gate_exps.weight", kExperts * kFFE, kHidden);
    L.up = quant(b + "ffn_up_exps.weight", kExperts * kFFE, kHidden);
    L.down = quant(b + "ffn_down_exps.weight", kExperts * kHidden, kFFE);
    for (const TView* t : {&L.q, &L.k, &L.o, &L.gate, &L.up})
      if (t->kind != kQ4) {
        fprintf(stderr, "gguf: layer %d: Q, K, O, gate and up must be Q4_K\n", l);
        exit(1);
      }
    L.router = f32(b + "ffn_gate_inp.weight", (size_t)kExperts * kHidden);
    L.attn_norm = f32(b + "attn_norm.weight", kHidden);
    L.ffn_norm = f32(b + "ffn_norm.weight", kHidden);
    L.q_norm = f32(b + "attn_q_norm.weight", kDim);
    L.k_norm = f32(b + "attn_k_norm.weight", kDim);
  }
  fm.emb = quant("token_embd.weight", 0, kHidden);
  if (fm.emb.kind != kQ4) {
    fprintf(stderr, "gguf: token_embd must be Q4_K\n");
    exit(1);
  }
  fm.vocab = fm.emb.M;
  fm.lm = quant("output.weight", fm.vocab, kHidden);
  if (fm.lm.kind != kQ6 || fm.vocab % 32) {
    fprintf(stderr, "gguf: output.weight must be Q6_K with a vocab divisible by 32\n");
    exit(1);
  }
  fm.out_norm = f32("output_norm.weight", kHidden);
}

static void random_model(FullModel& fm, int n_layers, int vocab, Rng& rng, float dlo, float dhi) {
  auto quant = [&](Kind kind, int M, int K) {
    const int bb = (int)block_bytes(kind);
    std::vector<uint8_t> buf((size_t)M * (K / 256) * bb);
    for (size_t i = 0; i + 8 <= buf.size(); i += 8) {
      uint64_t r = rng.next();
      memcpy(&buf[i], &r, 8);
    }
    for (size_t b = 0; b < buf.size() / bb; b++) {
      uint8_t* blk = &buf[b * bb];
      const uint16_t d = f2h(rng.uniform(dlo, dhi)), dm = f2h(rng.uniform(dlo, dhi));
      if (kind == kQ4) {
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dm, 2);
      } else {
        memcpy(blk + 208, &d, 2);
      }
    }
    fm.owned.push_back(std::move(buf));
    return TView{kind, M, K, fm.owned.back().data()};
  };
  auto f32 = [&](size_t n, float lo, float hi) {
    std::vector<uint8_t> buf(n * 4);
    float* f = (float*)buf.data();
    for (size_t i = 0; i < n; i++) f[i] = rng.uniform(lo, hi);
    fm.owned.push_back(std::move(buf));
    return (const float*)fm.owned.back().data();
  };
  fm.owned.reserve(32 * n_layers + 8);
  fm.n_layers = n_layers;
  fm.vocab = vocab;
  fm.layers.resize(n_layers);
  for (int l = 0; l < n_layers; l++) {
    FullLayer& L = fm.layers[l];
    const Kind vk = l == 0 ? kQ6 : kQ4;  // one layer of each kind
    L.q = quant(kQ4, kHeads * kDim, kHidden);
    L.k = quant(kQ4, kKV * kDim, kHidden);
    L.v = quant(vk, kKV * kDim, kHidden);
    L.o = quant(kQ4, kHidden, kHeads * kDim);
    L.gate = quant(kQ4, kExperts * kFFE, kHidden);
    L.up = quant(kQ4, kExperts * kFFE, kHidden);
    L.down = quant(vk, kExperts * kHidden, kFFE);
    L.router = f32((size_t)kExperts * kHidden, -0.05f, 0.05f);
    L.attn_norm = f32(kHidden, 0.8f, 1.2f);
    L.ffn_norm = f32(kHidden, 0.8f, 1.2f);
    L.q_norm = f32(kDim, 0.8f, 1.2f);
    L.k_norm = f32(kDim, 0.8f, 1.2f);
  }
  fm.emb = quant(kQ4, vocab, kHidden);
  fm.lm = quant(kQ6, vocab, kHidden);
  fm.out_norm = f32(kHidden, 0.8f, 1.2f);
}

// ------------------------------------------------------------------ shard
struct HostLayer {
  Tensor q, k, v, o, gate, up, down;
  std::vector<float> router, attn_norm, ffn_norm, q_norm, k_norm;
};

struct State {
  float *x, *q, *k, *v, *router_part, *router_ss, *act, *logits, *part_acc, *part_m, *part_l, *attn, *best_v;
  float* trace = nullptr;
  int *best_i, *token, *tokens_out, *trace_sel = nullptr;
  uint16_t *kbase, *vbase;  // [layer][max_ctx][kKVS][kDim]
  uint16_t **kcache, **vcache;
  unsigned *bar_count, *attn_cnt;
  unsigned long long* stamps;
  unsigned* status;
  Xchg* xc = nullptr;
#if MK_STAGE_SPLIT
  Xchg* stage_xc = nullptr;
#endif
  hipStream_t stream = nullptr;
  unsigned gen = 0, xseq = 0;
  int max_ctx = 0;
};

struct Shard {
  int rank = 0, device = 0;
  int n_layers = 0, vocab = 0, vocab_s = 0, vocab_off = 0, act_off = 0;
  std::vector<HostLayer> layers;
  std::vector<uint8_t> emb;
  Tensor lm;
  std::vector<float> out_norm;
  Layer* d_layers = nullptr;
  uint8_t* d_emb = nullptr;
  float* d_out_norm = nullptr;
  size_t weight_bytes = 0, token_bytes = 0;
  State s;
};

static float* to_dev(const float* h, size_t n) {
  float* p;
  CHECK(hipMalloc(&p, n * 4));
  CHECK(hipMemcpy(p, h, n * 4, hipMemcpyHostToDevice));
  return p;
}

static void shard_meta(Shard& sh, int rank, int device, int n_layers, int vocab) {
  sh.rank = rank;
  sh.device = device;
  sh.n_layers = n_layers;
  sh.vocab = vocab;
  const int vb = vocab / 32;  // vocab row blocks, split as evenly as they go
  sh.vocab_off = rank * vb / kShards * 32;
  sh.vocab_s = (rank + 1) * vb / kShards * 32 - sh.vocab_off;
  sh.act_off = rank * kFFS - down_first_sb(rank) * 256;
#ifdef MK_LAYOUT_LLAMA
  static_assert(kFFS % 256 == 0, "the role changes per layer, so the expert slice must start on a superblock");
#endif
  sh.layers.resize(n_layers);
  sh.token_bytes = (size_t)kHSB * kQ4Bytes + kHidden * 4;
}

// Shard rank's heads in a layer (first group, count) and its expert slice: llama.cpp's in MK_LAYOUT_LLAMA.
static void layer_slices(int rank, int layer, int& g0, int& ng, int& e) {
  const int r = layer_role(rank, layer);
  ng = grp_count(r);
#ifdef MK_LAYOUT_LLAMA
  g0 = llama_grp_first(rank, layer);
  e = rank;
#else
  g0 = grp_first(r);
  e = r;
#endif
}

static void slice_layer(HostLayer& L, const FullLayer& F, int rank, int layer) {
  auto vec = [](const float* p, size_t n) { return std::vector<float>(p, p + n); };
  int g0, ng, e;
  layer_slices(rank, layer, g0, ng, e);
  const int h0 = 8 * g0, nh = 8 * ng, kv0 = g0 / kIG, nkv = kv_count(layer_role(rank, layer));
  L.q = pad_rows(rows_of(F.q, h0 * kDim, nh * kDim), kQS);
  L.k = pad_rows(rows_of(F.k, kv0 * kDim, nkv * kDim), kKVR);
  L.v = pad_rows(rows_of(F.v, kv0 * kDim, nkv * kDim), kKVR);
  L.o = pad_cols(cols_of(F.o, h0 * kDim / 256, nh * kDim / 256), kQSB);
  L.gate = expert_rows(F.gate, kFFE, e * kFFS, kFFS);
  L.up = expert_rows(F.up, kFFE, e * kFFS, kFFS);
  L.down = down_slice(F.down, e);
  L.router = vec(F.router, (size_t)kExperts * kHidden);
  L.attn_norm = vec(F.attn_norm, kHidden);
  L.ffn_norm = vec(F.ffn_norm, kHidden);
  L.q_norm = vec(F.q_norm, kDim);
  L.k_norm = vec(F.k_norm, kDim);
}

static void upload_layer(Shard& sh, HostLayer& L, std::vector<Layer>& dl, bool keep_host) {
  size_t dense = (L.router.size() + 2 * kHidden + 2 * kDim) * 4, experts = 0;
  for (Tensor* t : {&L.q, &L.k, &L.v, &L.o}) dense += tensor_bytes(*t);
  for (Tensor* t : {&L.gate, &L.up, &L.down}) experts += tensor_bytes(*t);
  sh.weight_bytes += dense + experts;
  sh.token_bytes += dense + experts / kExperts * kTopK;
  for (Tensor* t : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) upload(*t, keep_host);
  dl.push_back({L.q.dev, L.k.dev, L.v.dev, L.v.dev_d, L.o.dev, to_dev(L.router.data(), L.router.size()), L.gate.dev,
                L.up.dev, L.down.dev, L.down.dev_d, L.down.kind == kQ6, to_dev(L.attn_norm.data(), kHidden),
                to_dev(L.ffn_norm.data(), kHidden), to_dev(L.q_norm.data(), kDim), to_dev(L.k_norm.data(), kDim)});
  if (!keep_host) {
    std::vector<float>().swap(L.router);
  }
}

// sh.emb, sh.lm and sh.out_norm are on the host; uploads them and the layer table.
static void finish_shard(Shard& sh, const std::vector<Layer>& dl, bool keep_host) {
  sh.weight_bytes += tensor_bytes(sh.lm) + sh.emb.size();
  sh.token_bytes += tensor_bytes(sh.lm);
  upload(sh.lm, keep_host);
  CHECK(hipMalloc(&sh.d_emb, sh.emb.size()));
  CHECK(hipMemcpy(sh.d_emb, sh.emb.data(), sh.emb.size(), hipMemcpyHostToDevice));
  if (!keep_host) std::vector<uint8_t>().swap(sh.emb);
  sh.d_out_norm = to_dev(sh.out_norm.data(), kHidden);
  CHECK(hipMalloc(&sh.d_layers, dl.size() * sizeof(Layer)));
  CHECK(hipMemcpy(sh.d_layers, dl.data(), dl.size() * sizeof(Layer), hipMemcpyHostToDevice));
}

static void build_shard(Shard& sh, const FullModel& fm, int rank, int device, bool keep_host) {
  CHECK(hipSetDevice(device));
  shard_meta(sh, rank, device, fm.n_layers, fm.vocab);
  std::vector<Layer> dl;
  for (int l = 0; l < fm.n_layers; l++) {
    slice_layer(sh.layers[l], fm.layers[l], rank, l);
    upload_layer(sh, sh.layers[l], dl, keep_host);
  }
  sh.emb.assign(fm.emb.data, fm.emb.data + (size_t)fm.emb.M * fm.emb.row_bytes());
  sh.lm = rows_of(fm.lm, sh.vocab_off, sh.vocab_s);
  sh.out_norm.assign(fm.out_norm, fm.out_norm + kHidden);
  finish_shard(sh, dl, keep_host);
}

// ----------------------------------------------------------- slice files
// One rank's slice as build_shard makes it, so a box without the GGUF reads only its own bytes (48 GB of the
// 235B's 142 at 3 shards): a header, per layer q k v o gate up down (kind, M, K, blocks) and router, attn / ffn /
// q / k norms (count, floats), then the embedding, this rank's LM head rows, the output norm, and a footer with a
// checksum of everything before it. `dump` writes one; `run` reads it when the model path ends in .mkslice.
struct SliceHeader {
  char magic[8];  // MKSLICE1
  char model[32];
  int32_t shards, rank, n_layers, vocab;
};
struct SliceFooter {
  char magic[8];  // MKSLEND1
  uint64_t sum;
};

struct SliceIO {
  FILE* f;
  uint64_t sum = 1469598103934665603ull;
  void hash(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      uint64_t w;
      memcpy(&w, b + i, 8);
      sum = (sum ^ w) * 1099511628211ull;
    }
    for (; i < n; i++) sum = (sum ^ b[i]) * 1099511628211ull;
  }
  void put(const void* p, size_t n) {
    hash(p, n);
    if (fwrite(p, 1, n, f) != n) die("write");
  }
  void get(void* p, size_t n) {
    if (fread(p, 1, n, f) != n) die("short read (truncated slice?)");
    hash(p, n);
  }
  [[noreturn]] static void die(const char* what) {
    fprintf(stderr, "slice: %s\n", what);
    exit(1);
  }
  void put_tensor(const Tensor& t) {
    const int32_t h[3] = {(int32_t)t.kind, t.M, t.K};
    put(h, sizeof(h));
    put(t.packed.data(), t.packed.size());
  }
  Tensor get_tensor() {
    int32_t h[3];
    get(h, sizeof(h));
    Tensor t{(Kind)h[0], h[1], h[2]};
    t.packed.resize(tensor_bytes(t));
    get(t.packed.data(), t.packed.size());
    return t;
  }
  void put_floats(const std::vector<float>& v) {
    const uint64_t n = v.size();
    put(&n, 8);
    put(v.data(), n * 4);
  }
  std::vector<float> get_floats() {
    uint64_t n;
    get(&n, 8);
    std::vector<float> v(n);
    get(v.data(), n * 4);
    return v;
  }
};

static int run_dump(const char* gguf, int rank, const char* out, int layers) {
  if (rank < 0 || rank >= kShards) {
    fprintf(stderr, "rank %d of %d\n", rank, kShards);
    return 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  FullModel fm;
  model_from_gguf(fm, gguf, layers);
  Shard sh;
  shard_meta(sh, rank, 0, fm.n_layers, fm.vocab);
  SliceIO io{strcmp(out, "-") ? fopen(out, "wb") : stdout};
  if (!io.f) SliceIO::die("cannot open the output");
  SliceHeader h{};
  memcpy(h.magic, "MKSLICE1", 8);
  snprintf(h.model, sizeof(h.model), "%s", kModelName);
  h.shards = kShards;
  h.rank = rank;
  h.n_layers = fm.n_layers;
  h.vocab = fm.vocab;
  io.put(&h, sizeof(h));
  size_t bytes = 0;
  for (int l = 0; l < fm.n_layers; l++) {
    HostLayer L;
    slice_layer(L, fm.layers[l], rank, l);
    for (Tensor* t : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) {
      io.put_tensor(*t);
      bytes += t->packed.size();
    }
    for (auto* v : {&L.router, &L.attn_norm, &L.ffn_norm, &L.q_norm, &L.k_norm}) io.put_floats(*v);
  }
  Tensor emb{fm.emb.kind, fm.emb.M, fm.emb.K};
  emb.packed.assign(fm.emb.data, fm.emb.data + (size_t)fm.emb.M * fm.emb.row_bytes());
  io.put_tensor(emb);
  io.put_tensor(rows_of(fm.lm, sh.vocab_off, sh.vocab_s));
  io.put_floats(std::vector<float>(fm.out_norm, fm.out_norm + kHidden));
  SliceFooter ft{};
  memcpy(ft.magic, "MKSLEND1", 8);
  ft.sum = io.sum;
  if (fwrite(&ft, sizeof(ft), 1, io.f) != 1 || fflush(io.f) != 0) SliceIO::die("write");
  if (io.f != stdout) fclose(io.f);
  fprintf(stderr, "rank %d of %d, %d layers: %.2f GB of layer tensors written in %.1f s, checksum %016llx\n", rank,
          kShards, fm.n_layers, bytes / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
          (unsigned long long)ft.sum);
  return 0;
}

// The first `layers` layers of a slice; the checksum is checked when the whole file is read.
static void load_slice(Shard& sh, const char* path, int rank, int device, int layers) {
  SliceIO io{fopen(path, "rb")};
  if (!io.f) SliceIO::die("cannot open the slice");
  SliceHeader h;
  io.get(&h, sizeof(h));
  if (memcmp(h.magic, "MKSLICE1", 8) || strncmp(h.model, kModelName, sizeof(h.model)) || h.shards != kShards ||
      h.rank != rank) {
    fprintf(stderr, "slice: %s is %.32s, rank %d of %d; this build is %s, rank %d of %d\n", path, h.model, h.rank,
            h.shards, kModelName, rank, kShards);
    exit(1);
  }
  CHECK(hipSetDevice(device));
  const int n = std::min(layers, (int)h.n_layers);
  shard_meta(sh, rank, device, n, h.vocab);
  std::vector<Layer> dl;
  for (int l = 0; l < h.n_layers; l++) {
    HostLayer L;
    for (Tensor* t : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) *t = io.get_tensor();
    for (auto* v : {&L.router, &L.attn_norm, &L.ffn_norm, &L.q_norm, &L.k_norm}) *v = io.get_floats();
    if (l < n) {
      sh.layers[l] = std::move(L);
      upload_layer(sh, sh.layers[l], dl, false);
    }
  }
  sh.emb = io.get_tensor().packed;
  sh.lm = io.get_tensor();
  sh.out_norm = io.get_floats();
  SliceFooter ft;
  const uint64_t sum = io.sum;
  if (fread(&ft, sizeof(ft), 1, io.f) != 1 || memcmp(ft.magic, "MKSLEND1", 8)) SliceIO::die("no footer (truncated?)");
  if (ft.sum != sum) SliceIO::die("checksum mismatch: the copy is damaged");
  fclose(io.f);
  finish_shard(sh, dl, false);
}

constexpr int kMaxSteps = 4096;

// k_ext / v_ext: per-layer caches owned by the caller (llama.cpp's), max_ctx cells each; else allocated here.
static void alloc_state(Shard& sh, int max_ctx, bool trace, const std::vector<uint16_t*>* k_ext = nullptr,
                        const std::vector<uint16_t*>* v_ext = nullptr) {
  CHECK(hipSetDevice(sh.device));
  State& s = sh.s;
  int per_cu = 0, cus = 0;
  CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, sh.device));
#if MK_STAGE_SPLIT
  for (const void* kernel : {(const void*)decode_attention_stage, (const void*)decode_moe_stage,
                             (const void*)decode_head_stage}) {
#else
  for (const void* kernel : {(const void*)decode}) {
#endif
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, kBlock, 0));
    if (kGrid > per_cu * cus) {
      fprintf(stderr, "grid %d: %d per CU x %d CUs fit\n", kGrid, per_cu, cus);
      exit(1);
    }
  }
  s.max_ctx = max_ctx;
  auto f = [](float*& p, size_t n) {
    CHECK(hipMalloc(&p, n * 4));
    CHECK(hipMemset(p, 0, n * 4));
  };
  f(s.x, kHidden);
  f(s.q, kQS);
  f(s.k, kKVR);
  f(s.v, kKVR);
  f(s.router_part, (size_t)kExperts * kHB);
  f(s.router_ss, kHB);
  f(s.act, (size_t)kTopK * kFFP);
  f(s.logits, sh.vocab_s);
  f(s.part_acc, (size_t)kHS * kMaxSplits * kDim);
  f(s.part_m, (size_t)kHS * kMaxSplits);
  f(s.part_l, (size_t)kHS * kMaxSplits);
  f(s.attn, (size_t)kOQG * kWaves * 256);
  f(s.best_v, kGrid);
  CHECK(hipMalloc(&s.attn_cnt, kGroups * 4));
  CHECK(hipMemset(s.attn_cnt, 0, kGroups * 4));
  CHECK(hipMalloc(&s.best_i, kGrid * 4));
  CHECK(hipMalloc(&s.token, 4));
  CHECK(hipMalloc(&s.tokens_out, kMaxSteps * 4));
  {
    std::vector<uint16_t*> kp(sh.n_layers), vp(sh.n_layers);
    if (k_ext) {
      s.kbase = s.vbase = nullptr;
      kp = *k_ext;
      vp = *v_ext;
    } else {
      const size_t kv = (size_t)sh.n_layers * max_ctx * kKVS * kDim * 2;
      CHECK(hipMalloc(&s.kbase, kv));
      CHECK(hipMalloc(&s.vbase, kv));
      CHECK(hipMemset(s.kbase, 0, kv));
      CHECK(hipMemset(s.vbase, 0, kv));
      for (int l = 0; l < sh.n_layers; l++) {
        kp[l] = s.kbase + (size_t)l * max_ctx * kKVS * kDim;
        vp[l] = s.vbase + (size_t)l * max_ctx * kKVS * kDim;
      }
    }
    CHECK(hipMalloc(&s.kcache, sh.n_layers * sizeof(uint16_t*)));
    CHECK(hipMalloc(&s.vcache, sh.n_layers * sizeof(uint16_t*)));
    CHECK(hipMemcpy(s.kcache, kp.data(), kp.size() * sizeof(uint16_t*), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(s.vcache, vp.data(), vp.size() * sizeof(uint16_t*), hipMemcpyHostToDevice));
  }
  CHECK(hipMalloc(&s.bar_count, 8));
  CHECK(hipMemset(s.bar_count, 0, 8));
  CHECK(hipMalloc(&s.stamps, kMaxStamps * 8));
  CHECK(hipHostMalloc((void**)&s.status, 64, hipHostMallocCoherent | hipHostMallocPortable));
  if (kShards > 1) {
    CHECK(hipHostMalloc((void**)&s.xc, sizeof(Xchg), hipHostMallocCoherent | hipHostMallocPortable));
    memset(s.xc, 0, sizeof(Xchg));
#if MK_STAGE_SPLIT
    CHECK(hipMalloc(&s.stage_xc, sizeof(Xchg)));
    CHECK(hipMemset(s.stage_xc, 0, sizeof(Xchg)));
#endif
  }
  if (trace) {
    f(s.trace, (size_t)(sh.n_layers + 1) * kHidden);
    CHECK(hipMalloc(&s.trace_sel, (size_t)sh.n_layers * kTopK * 4));
  }
  CHECK(hipStreamCreateWithFlags(&s.stream, hipStreamNonBlocking));
}

static Args make_args(const Shard& sh, int pos, int n_steps, bool stamps, const int* inputs, float* logits_all) {
  const State& s = sh.s;
  Args a{};
  a.n_layers = sh.n_layers;
  a.vocab = sh.vocab_s;
  a.vocab_off = sh.vocab_off;
  a.rank = sh.rank;
  a.act_off = sh.act_off;
  a.pos = pos;
  a.n_steps = n_steps;
  a.max_ctx = s.max_ctx;
  a.gen_base = s.gen;
  a.xseq_base = s.xseq;
  a.layers = sh.d_layers;
  a.emb = sh.d_emb;
  a.lm = sh.lm.dev;
  a.lm_d = sh.lm.dev_d;
  a.out_norm = sh.d_out_norm;
  a.x = s.x;
  a.q = s.q;
  a.k = s.k;
  a.v = s.v;
  a.router_part = s.router_part;
  a.router_ss = s.router_ss;
  a.act = s.act;
  a.logits = s.logits;
  a.part_acc = s.part_acc;
  a.part_m = s.part_m;
  a.part_l = s.part_l;
  a.attn = s.attn;
  a.attn_cnt = s.attn_cnt;
  a.kcache = s.kcache;
  a.vcache = s.vcache;
  a.trace = s.trace;
  a.trace_sel = s.trace_sel;
  a.best_v = s.best_v;
  a.best_i = s.best_i;
  a.token = s.token;
  a.tokens_out = s.tokens_out;
  a.inputs = inputs;
  a.logits_all = logits_all;
  a.bar_count = s.bar_count;
  a.stamps = stamps ? s.stamps : nullptr;
  a.status = s.status;
#if MK_STAGE_SPLIT
  a.xc = s.stage_xc;
#else
  a.xc = s.xc;
#endif
  return a;
}

// ------------------------------------------------------------------ proxy
// The slowest exchange of the last proxy() call: time from the request to the peers' answer, and its index.
struct ProxyStats {
  double max_us = 0;
  unsigned max_at = 0;
};

struct ExchangeScratch {
  std::vector<float> buf = std::vector<float>(kHidden), full;
  std::vector<uint16_t> half;
};

static void exchange_host(Xchg* xc, Transport* tp, unsigned op, unsigned n, ExchangeScratch& scratch) {
  static const bool half = getenv("MK_XCHG16") && atoi(getenv("MK_XCHG16"));
  if (op == kOpSum) {
    memcpy(scratch.buf.data(), xc->out, n * 4);
    if (half) tp->allreduce_sum16(scratch.buf.data(), n, scratch.half);
    else tp->allreduce_sum(scratch.buf.data(), n, scratch.full);
    memcpy(xc->in, scratch.buf.data(), n * 4);
  } else {
    float mine[2] = {xc->out[0], xc->out[1]};
    std::vector<float> all(2 * tp->size);
    tp->all_gather(mine, all.data(), 8);
    float best = all[0];
    int bidx = __builtin_bit_cast(int, all[1]);
    for (int p = 1; p < tp->size; p++) {
      const float v = all[2 * p];
      const int i = __builtin_bit_cast(int, all[2 * p + 1]);
      if (v > best || (v == best && i < bidx)) {
        best = v;
        bidx = i;
      }
    }
    xc->in[0] = best;
    xc->in[1] = __builtin_bit_cast(float, bidx);
  }
}

// The host side of one shard's exchanges: count of them, numbered after first.
static bool proxy(Xchg* xc, Transport* tp, unsigned first, unsigned count, ProxyStats* st = nullptr) {
  ExchangeScratch scratch;
  for (unsigned k = 1; k <= count; k++) {
    const unsigned seq = first + k;
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (__atomic_load_n(&xc->out_seq, __ATOMIC_ACQUIRE) != seq)
      if (std::chrono::steady_clock::now() > give_up) {
        fprintf(stderr, "proxy rank %d: no exchange %u from the GPU\n", tp->rank, seq);
        return false;
      }
    const unsigned op = __atomic_load_n(&xc->op, __ATOMIC_RELAXED), n = __atomic_load_n(&xc->n, __ATOMIC_RELAXED);
    const double t0 = st ? now_us() : 0;
    exchange_host(xc, tp, op, n, scratch);
    if (st && now_us() - t0 > st->max_us) {
      st->max_us = now_us() - t0;
      st->max_at = k;
    }
    __atomic_store_n(&xc->in_seq, seq, __ATOMIC_RELEASE);
  }
  return true;
}

// Decodes n_steps on every shard at once, each with its exchanges served by its own transport.
struct Run {
  std::vector<Shard*> shards;
  std::vector<Transport*> tps;  // one per shard, null with one shard
};

#if MK_STAGE_SPLIT
static bool launch_shard_staged(Shard& sh, Transport* tp, int pos, int n_steps, bool stamps,
                                const int* inputs, float* logits_all, ProxyStats* st) {
  if (kShards <= 1 || !tp || sh.n_layers <= 0 || stamps || n_steps <= 0 ||
      n_steps > kMaxSteps || pos + n_steps > sh.s.max_ctx) {
    fprintf(stderr, "staged unsupported launch dimensions or timing-stamp request\n");
    return false;
  }
  CHECK(hipSetDevice(sh.device));
  State& s = sh.s;
  ExchangeScratch scratch;
  unsigned exchange_index = 0;
  for (int step = 0; step < n_steps; ++step) {
    auto args = [&] {
      Args a = make_args(sh, pos + step, 1, false, inputs ? inputs + step : nullptr,
                         logits_all ? logits_all + (size_t)step * sh.vocab_s : nullptr);
      if (step != 0) a.trace = nullptr;
      return a;
    };
    auto finish = [&](unsigned barriers, unsigned op, unsigned n) {
      CHECK(hipMemcpyAsync(s.xc->out, s.stage_xc->out, n * 4, hipMemcpyDeviceToHost, s.stream));
      CHECK(hipMemcpyAsync(s.status, s.bar_count, 8, hipMemcpyDeviceToHost, s.stream));
      CHECK(hipStreamSynchronize(s.stream));
      s.gen += barriers;
      if (s.status[1] || s.status[0] != s.gen * kGrid) {
        fprintf(stderr, "staged rank %d: stage %u has %u arrivals, expected %u, error=%u\n",
                sh.rank, exchange_index + 1, s.status[0], s.gen * kGrid, s.status[1]);
        exit(1);
      }
      const double t0 = st ? now_us() : 0;
      exchange_host(s.xc, tp, op, n, scratch);
      ++s.xseq;
      ++exchange_index;
      if (st && now_us() - t0 > st->max_us) {
        st->max_us = now_us() - t0;
        st->max_at = exchange_index;
      }
      if (op == kOpSum)
        CHECK(hipMemcpyAsync(s.stage_xc->in, s.xc->in, n * 4, hipMemcpyHostToDevice, s.stream));
    };
    for (int layer = 0; layer < sh.n_layers; ++layer) {
      hipLaunchKernelGGL(decode_attention_stage, dim3(kGrid), dim3(kBlock), 0, s.stream, args(), layer);
      CHECK(hipGetLastError());
      finish(3, kOpSum, kHidden);
      hipLaunchKernelGGL(decode_moe_stage, dim3(kGrid), dim3(kBlock), 0, s.stream, args(), layer);
      CHECK(hipGetLastError());
      finish(2, kOpSum, kHidden);
    }
    hipLaunchKernelGGL(decode_head_stage, dim3(kGrid), dim3(kBlock), 0, s.stream, args());
    CHECK(hipGetLastError());
    finish(2, kOpArgmax, 2);
    const int token = __builtin_bit_cast(int, s.xc->in[1]);
    CHECK(hipMemcpy(s.token, &token, 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(s.tokens_out + step, &token, 4, hipMemcpyHostToDevice));
  }
  return true;
}

static bool launch_steps_staged(Run& run, int pos, int n_steps, bool stamps,
                                const std::vector<int*>& inputs,
                                const std::vector<float*>& logits_all, ProxyStats* st) {
  auto one = [&](size_t i) {
    return launch_shard_staged(*run.shards[i], run.tps[i], pos, n_steps, stamps,
                               inputs.empty() ? nullptr : inputs[i],
                               logits_all.empty() ? nullptr : logits_all[i], st);
  };
  if (run.shards.size() == 1) return one(0);
  std::vector<std::thread> threads;
  std::vector<int> ok(run.shards.size(), 0);
  for (size_t i = 0; i < run.shards.size(); ++i) threads.emplace_back([&, i] { ok[i] = one(i); });
  for (auto& thread : threads) thread.join();
  return std::all_of(ok.begin(), ok.end(), [](int value) { return value != 0; });
}
#endif

// false (after a message) when a step failed: watchdog, exchange, barrier count, or buffers too small.
static bool launch_steps(Run& run, int pos, int n_steps, bool stamps, const std::vector<int*>& inputs = {},
                         const std::vector<float*>& logits_all = {}, ProxyStats* st = nullptr) {
#if MK_STAGE_SPLIT
  return launch_steps_staged(run, pos, n_steps, stamps, inputs, logits_all, st);
#endif
  for (size_t i = 0; i < run.shards.size(); i++) {
    Shard& sh = *run.shards[i];
    if (n_steps > kMaxSteps || pos + n_steps > sh.s.max_ctx) {
      fprintf(stderr, "launch: %d steps from %d exceed the buffers\n", n_steps, pos);
      return false;
    }
    CHECK(hipSetDevice(sh.device));
    hipLaunchKernelGGL(decode, dim3(kGrid), dim3(kBlock), 0, sh.s.stream,
                       make_args(sh, pos, n_steps, stamps, inputs.empty() ? nullptr : inputs[i],
                                 logits_all.empty() ? nullptr : logits_all[i]));
    CHECK(hipGetLastError());
  }
  const unsigned count = (unsigned)n_steps * exchanges_per_step(run.shards[0]->n_layers);
  bool ok = true;
  if (kShards > 1) {
    std::vector<std::thread> th;
    std::vector<int> res(run.shards.size(), 1);
    for (size_t i = 0; i < run.shards.size(); i++)
      th.emplace_back([&, i] { res[i] = proxy(run.shards[i]->s.xc, run.tps[i], run.shards[i]->s.xseq, count, st); });
    for (auto& t : th) t.join();
    for (int r : res) ok &= r != 0;
  }
  for (Shard* sh : run.shards) {
    CHECK(hipSetDevice(sh->device));
    CHECK(hipStreamSynchronize(sh->s.stream));
    unsigned flags[2];
    CHECK(hipMemcpy(flags, sh->s.bar_count, 8, hipMemcpyDeviceToHost));
    if (flags[1]) ok = false;
    sh->s.gen += (unsigned)n_steps * barriers_per_step(sh->n_layers, kShards > 1);
    sh->s.xseq += kShards > 1 ? count : 0;
    if (flags[0] != sh->s.gen * kGrid) {
      fprintf(stderr, "rank %d: %u barrier arrivals, expected %u\n", sh->rank, flags[0], sh->s.gen * kGrid);
      ok = false;
    }
  }
  if (!ok) fprintf(stderr, "decode failed (watchdog or exchange)\n");
  return ok;
}

static void launch_all(Run& run, int pos, int n_steps, bool stamps, const std::vector<int*>& inputs = {},
                       const std::vector<float*>& logits_all = {}) {
  if (!launch_steps(run, pos, n_steps, stamps, inputs, logits_all)) exit(1);
}

// --------------------------------------------------------- CPU reference
// The shards' arithmetic in the kernel's reduction orders, partials added in rank order; except expf/cosf.
static void cpu_rmsnorm(std::vector<float>& v, const float* w) {
  const int n = (int)v.size(), groups_per_wave = n / 256 / kWaves;
  float red[kWaves];
  for (int wv = 0; wv < kWaves; wv++) {
    float lanes[32];
    for (int l = 0; l < 32; l++) {
      float ss = 0.f;
      for (int g = 0; g < groups_per_wave; g++)
        for (int e = 0; e < 8; e++) {
          const float x = v[(wv + kWaves * g) * 256 + 8 * l + e];
          ss = std::fmaf(x, x, ss);
        }
      lanes[l] = ss;
    }
    red[wv] = wave_tree_sum(lanes);
  }
  float tot = 0.f;
  for (int k = 0; k < kWaves; k++) tot += red[k];
  const float inv = 1.f / sqrtf(tot / (float)n + kEps);
  for (int e = 0; e < n; e++) v[e] = v[e] * inv * w[e];
}

struct Act {
  std::vector<int8_t> q;
  std::vector<int> s16;
  std::vector<float> d;
};

static Act cpu_quantize(const std::vector<float>& v) {
  const int n = (int)v.size();
  Act a;
  a.q.resize(n);
  a.s16.assign(n / 16, 0);
  a.d.resize(n / 256);
  for (int sb = 0; sb < n / 256; sb++) {
    float m = 0.f;
    for (int e = 0; e < 256; e++) m = std::max(m, fabsf(v[sb * 256 + e]));
    const float scale = m > 0.f ? m / 127.f : 1.f;
    a.d[sb] = scale;
    for (int e = 0; e < 256; e++) {
      int q = (int)rintf(v[sb * 256 + e] / scale);
      q = std::max(-127, std::min(127, q));
      a.q[sb * 256 + e] = (int8_t)q;
      a.s16[(sb * 256 + e) / 16] += q;
    }
  }
  return a;
}

static float cpu_q4_block(const uint8_t* b, const Act& x, int sb) {
  uint16_t dh, mh;
  memcpy(&dh, b, 2);
  memcpy(&mh, b + 2, 2);
  int sc[8], mn[8];
  q4_scales(b + 4, sc, mn);
  int isc = 0, imn = 0;
  for (int s = 0; s < 8; s++) {
    const int j = s >> 1, hi = s & 1;
    int sumi = 0;
    for (int l = 0; l < 32; l++) {
      const uint8_t byte = b[16 + 32 * j + l];
      sumi += (hi ? byte >> 4 : byte & 15) * (int)x.q[sb * 256 + 64 * j + 32 * hi + l];
    }
    isc += sc[s] * sumi;
    imn += mn[s] * (x.s16[sb * 16 + 2 * s] + x.s16[sb * 16 + 2 * s + 1]);
  }
  return std::fmaf(h2f(dh), (float)isc, -(h2f(mh) * (float)imn));
}

static int q6_quant(const uint8_t* b, int kk, int& sc) {
  const int h = kk >> 7, r = kk & 127, j = r >> 5, l = r & 31;
  const int qlb = b[64 * h + 32 * (j & 1) + l];
  const int ql = (j & 2) ? qlb >> 4 : qlb & 15;
  const int qh = (b[128 + 32 * h + l] >> (2 * j)) & 3;
  sc = (int)(int8_t)b[192 + 8 * h + (l >> 4) + 2 * j];
  return ql | (qh << 4);
}

static float cpu_q6_block(const uint8_t* b, const Act& x, int sb) {
  uint16_t dh;
  memcpy(&dh, b + 208, 2);
  int sumi[16] = {0}, scs[16] = {0};
  for (int kk = 0; kk < 256; kk++) {
    int sc;
    const int q = q6_quant(b, kk, sc);
    sumi[kk / 16] += q * (int)x.q[sb * 256 + kk];
    scs[kk / 16] = sc;
  }
  int isum = 0;
  for (int s = 0; s < 16; s++) isum += scs[s] * (sumi[s] - 32 * x.s16[sb * 16 + s]);
  return h2f(dh) * (float)isum;
}

// split = waves sharing K: wave w takes superblocks w, w + split, ...; partials summed in wave order.
static float cpu_row(const Tensor& t, int m, const Act& x, int split) {
  const int SB = t.K / 256;
  const size_t bb = block_bytes(t.kind);
  float tot = 0.f;
  for (int w = 0; w < split; w++) {
    float acc = 0.f;
    for (int sb = w; sb < SB; sb += split) {
      const uint8_t* b = t.packed.data() + ((size_t)m * SB + sb) * bb;
      acc = std::fmaf(x.d[sb], t.kind == kQ4 ? cpu_q4_block(b, x, sb) : cpu_q6_block(b, x, sb), acc);
    }
    tot += acc;
  }
  return tot;
}

static void cpu_rope_tables(int pos, float* cs, float* sn) {
  const float theta_scale = powf(kTheta, -2.f / (float)kDim);
  for (int i = 0; i < kRot; i++) {
    float theta = (float)pos;
    for (int k = 0; k < i; k++) theta *= theta_scale;
    cs[i] = cosf(theta);
    sn[i] = sinf(theta);
  }
}

static void cpu_head_norm_rope(float* h, const float* w, const float* cs, const float* sn) {
  float lanes[32];
  for (int l = 0; l < 32; l++) {
    float s = h[l] * h[l];
    for (int j = 1; j < 4; j++) s = std::fmaf(h[l + 32 * j], h[l + 32 * j], s);
    lanes[l] = s;
  }
  const float inv = 1.f / sqrtf(wave_tree_sum(lanes) / (float)kDim + kEps);
  float y[kDim];
  for (int i = 0; i < kDim; i++) y[i] = h[i] * inv * w[i];
  for (int i = 0; i < kRot; i++) {
    h[i] = std::fmaf(y[i], cs[i], -(y[i + kRot] * sn[i]));
    h[i + kRot] = std::fmaf(y[i + kRot], cs[i], y[i] * sn[i]);
  }
}

static float cpu_q4_value_raw(const std::vector<uint8_t>& W, int m, int e) {
  const uint8_t* b = W.data() + ((size_t)m * kHSB + e / 256) * kQ4Bytes;
  const int kk = e % 256, j = kk / 64, hi = (kk / 32) & 1, l = kk % 32;
  int sc[8], mn[8];
  q4_scales(b + 4, sc, mn);
  const int s = 2 * j + hi;
  const int q = hi ? b[16 + 32 * j + l] >> 4 : b[16 + 32 * j + l] & 15;
  const float d = h2f((uint16_t)(b[0] | b[1] << 8)), dmin = h2f((uint16_t)(b[2] | b[3] << 8));
  return std::fmaf(d * (float)sc[s], (float)q, -(dmin * (float)mn[s]));
}

struct CpuKV {
  std::vector<uint16_t> kc, vc;
  int max_ctx;
  // a layer's region holds max_ctx * kKVS heads; positions are kv_stride(role) heads apart, as on the GPU
  size_t off(int layer, int pos, int g, int st) const {
    return (size_t)layer * max_ctx * kKVS * kDim + ((size_t)pos * st + g) * kDim;
  }
};

static void cpu_route(const float* lg, int* sel, float* w) {
  float m = -INFINITY;
  for (int e = 0; e < kExperts; e++) m = std::max(m, lg[e]);
  float ex[kExperts];
  for (int e = 0; e < kExperts; e++) ex[e] = expf(lg[e] - m);
  const float S = ((wave_tree_sum(ex) + wave_tree_sum(ex + 32)) + wave_tree_sum(ex + 64)) + wave_tree_sum(ex + 96);
  bool taken[kExperts] = {};
  float pk[kTopK], psum = 0.f;
  for (int k = 0; k < kTopK; k++) {
    int bi = -1;
    for (int e = 0; e < kExperts; e++)
      if (!taken[e] && (bi < 0 || lg[e] > lg[bi])) bi = e;
    taken[bi] = true;
    sel[k] = bi;
    pk[k] = ex[bi] / S;
  }
  for (int k = 0; k < kTopK; k++) psum += pk[k];
  for (int k = 0; k < kTopK; k++) w[k] = pk[k] / psum;
}

// x += the shards' partials: directly with one shard, else their sum in rank order.
static void cpu_add_partials(std::vector<float>& x, const std::vector<std::vector<float>>& part) {
  for (int m = 0; m < kHidden; m++) {
    float s = part[0][m];
    for (size_t r = 1; r < part.size(); r++) s += part[r][m];
    x[m] = x[m] + s;
  }
}

// Shard r's attention for this layer: q, k, v of its heads in, attention output out (zero for the padding
// heads); K, V of its real KV slots into its cache.
static void cpu_attention(CpuKV& cs, int rank, int layer, int pos, std::vector<float>& q, std::vector<float>& k,
                          const std::vector<float>& v, const HostLayer& L, std::vector<float>& attn) {
  float cosv[kRot], sinv[kRot];
  cpu_rope_tables(pos, cosv, sinv);
  const int ntok = pos + 1;
  int nchunk, chunk;
  shard_attn_split(ntok, nchunk, chunk);
  const float scale = 1.f / sqrtf((float)kDim);
  for (int h = 0; h < kHS; h++) cpu_head_norm_rope(&q[h * kDim], L.q_norm.data(), cosv, sinv);
  const int role = layer_role(rank, layer), st = kv_stride(role);
  for (int g = 0; g < kv_count(role); g++) {
    cpu_head_norm_rope(&k[g * kDim], L.k_norm.data(), cosv, sinv);
    for (int i = 0; i < kDim; i++) {
      cs.kc[cs.off(layer, pos, g, st) + i] = f2h(k[g * kDim + i]);
      cs.vc[cs.off(layer, pos, g, st) + i] = f2h(v[g * kDim + i]);
    }
  }
  std::vector<float> pm(kHS * kMaxSplits), pl(kHS * kMaxSplits), pa((size_t)kHS * kMaxSplits * kDim);
  const int nh = 8 * grp_count(role);
  for (int head = 0; head < nh; head++) {
    const int g = grp_slot(role, head / 8);
    for (int c = 0; c < nchunk; c++) {
      const int t0 = c * chunk, t1 = std::min(t0 + chunk, ntok), n = t1 - t0;
      std::vector<float> p(n);
      for (int j = 0; j < n; j++) {
        float parts[16];
        for (int pp = 0; pp < 16; pp++) {
          float s = 0.f;
          for (int e = 0; e < 8; e++)
            s = std::fmaf(q[head * kDim + 8 * pp + e], h2f(cs.kc[cs.off(layer, t0 + j, g, st) + 8 * pp + e]), s);
          parts[pp] = s;
        }
        for (int off = 8; off > 0; off >>= 1) {
          float nxt[16];
          for (int l = 0; l < 16; l++) nxt[l] = parts[l] + parts[l ^ off];
          memcpy(parts, nxt, sizeof(parts));
        }
        p[j] = parts[0] * scale;
      }
      float mx = -INFINITY;
      for (float s : p) mx = std::max(mx, s);
      float lanes[32];
      for (int l = 0; l < 32; l++) {
        float s = 0.f;
        for (int j = l; j < n; j += 32) s += (p[j] = expf(p[j] - mx));
        lanes[l] = s;
      }
      const float ls = wave_tree_sum(lanes);
      for (int i = 0; i < kDim; i++) {
        float acc[16];
        for (int sub = 0; sub < 16; sub++) {
          acc[sub] = 0.f;
          for (int j = sub; j < n; j += 16) acc[sub] = std::fmaf(p[j], h2f(cs.vc[cs.off(layer, t0 + j, g, st) + i]), acc[sub]);
        }
        float out = 0.f;
        for (int w = 0; w < kWaves; w++) out += acc[2 * w] + acc[2 * w + 1];
        pa[((size_t)head * kMaxSplits + c) * kDim + i] = out;
      }
      pm[head * kMaxSplits + c] = mx;
      pl[head * kMaxSplits + c] = ls;
    }
  }
  attn.assign(kQS, 0.f);
  for (int e = 0; e < nh * kDim; e++) {
    const int head = e / kDim, d = e % kDim;
    float M = -INFINITY;
    for (int c = 0; c < nchunk; c++) M = std::max(M, pm[head * kMaxSplits + c]);
    float l = 0.f, o = 0.f;
    for (int c = 0; c < nchunk; c++) {
      const float s = expf(pm[head * kMaxSplits + c] - M);
      l = std::fmaf(pl[head * kMaxSplits + c], s, l);
      o = std::fmaf(pa[((size_t)head * kMaxSplits + c) * kDim + d], s, o);
    }
    attn[e] = o / l;
  }
}

static int cpu_decode(const std::vector<Shard*>& sh, std::vector<CpuKV>& kv, int token, int pos,
                      std::vector<float>& trace, std::vector<float>& logits, std::vector<int>& sel_trace) {
  const int N = (int)sh.size(), nl = sh[0]->n_layers;
  std::vector<float> x(kHidden);
  for (int e = 0; e < kHidden; e++) x[e] = cpu_q4_value_raw(sh[0]->emb, token, e);
  trace.assign((size_t)(nl + 1) * kHidden, 0.f);
  sel_trace.assign((size_t)nl * kTopK, 0);
  std::vector<std::vector<float>> part(N, std::vector<float>(kHidden));
  for (int layer = 0; layer < nl; layer++) {
    const HostLayer& L0 = sh[0]->layers[layer];
    memcpy(&trace[(size_t)layer * kHidden], x.data(), kHidden * 4);
    std::vector<float> xn = x;
    cpu_rmsnorm(xn, L0.attn_norm.data());
    Act a = cpu_quantize(xn);
    for (int r = 0; r < N; r++) {
      const HostLayer& L = sh[r]->layers[layer];
      std::vector<float> q(kQS), k(kKVR), v(kKVR), attn;
      for (int m = 0; m < kQS; m++) q[m] = cpu_row(L.q, m, a, kWaves);
      for (int m = 0; m < kKVR; m++) k[m] = cpu_row(L.k, m, a, kWaves);
      for (int m = 0; m < kKVR; m++) v[m] = cpu_row(L.v, m, a, kWaves);
      cpu_attention(kv[r], sh[r]->rank, layer, pos, q, k, v, L, attn);
      const Act ao = cpu_quantize(attn);
      for (int m = 0; m < kHidden; m++) part[r][m] = cpu_row(L.o, m, ao, kOWR);
    }
    if (N == 1)
      for (int m = 0; m < kHidden; m++) x[m] = x[m] + part[0][m];
    else
      cpu_add_partials(x, part);

    float rl[kExperts];
    {
      std::vector<float> rp((size_t)kExperts * kHB);
      std::vector<float> ss(kHB);
      for (int b = 0; b < kHB; b++) {
        float xw[32], sq[32];
        for (int l = 0; l < 32; l++) {
          const int i = 32 * b + l;
          xw[l] = x[i] * L0.ffn_norm[i];
          sq[l] = x[i] * x[i];
        }
        ss[b] = wave_tree_sum(sq);
        for (int e = 0; e < kExperts; e++) {
          float h[2];
          for (int rh = 0; rh < 2; rh++) {
            float acc = 0.f;
            for (int j = 0; j < 16; j++)
              acc = std::fmaf(L0.router[(size_t)e * kHidden + 32 * b + 16 * rh + j], xw[16 * rh + j], acc);
            h[rh] = acc;
          }
          rp[(size_t)e * kHB + b] = h[0] + h[1];
        }
      }
      float lanes[32];
      for (int l = 0; l < 32; l++) {
        float s = ss[l];
        for (int j = 1; j < kHB / 32; j++) s += ss[l + 32 * j];
        lanes[l] = s;
      }
      const float inv = 1.f / sqrtf(wave_tree_sum(lanes) / (float)kHidden + kEps);
      for (int e = 0; e < kExperts; e++) {
        float tot = 0.f;
        for (int b = 0; b < kHB; b++) tot += rp[(size_t)e * kHB + b];
        rl[e] = tot * inv;
      }
    }
    xn = x;
    cpu_rmsnorm(xn, L0.ffn_norm.data());
    a = cpu_quantize(xn);
    int sel[kTopK];
    float selw[kTopK];
    cpu_route(rl, sel, selw);
    memcpy(&sel_trace[(size_t)layer * kTopK], sel, sizeof(sel));
    for (int r = 0; r < N; r++) {
      const HostLayer& L = sh[r]->layers[layer];
      std::vector<float> y((size_t)kTopK * kHidden);
      for (int s = 0; s < kTopK; s++) {
        std::vector<float> act(kFFP, 0.f);
        for (int i = 0; i < kFFS; i++) {
          const int row = sel[s] * kFFS + i;
          const float g = cpu_row(L.gate, row, a, kWaves), u = cpu_row(L.up, row, a, kWaves);
          act[sh[r]->act_off + i] = (g / (1.f + expf(-g))) * u;
        }
        const Act aq = cpu_quantize(act);
        for (int m = 0; m < kHidden; m++) y[(size_t)s * kHidden + m] = cpu_row(L.down, sel[s] * kHidden + m, aq, 1);
      }
      for (int m = 0; m < kHidden; m++) {
        float acc = 0.f;
        for (int s = 0; s < kTopK; s++) acc = std::fmaf(selw[s], y[(size_t)s * kHidden + m], acc);
        part[r][m] = acc;
      }
    }
    if (N == 1)
      for (int m = 0; m < kHidden; m++) x[m] += part[0][m];
    else
      cpu_add_partials(x, part);
  }
  memcpy(&trace[(size_t)nl * kHidden], x.data(), kHidden * 4);
  std::vector<float> xn = x;
  cpu_rmsnorm(xn, sh[0]->out_norm.data());
  const Act a = cpu_quantize(xn);
  logits.assign(sh[0]->vocab, 0.f);
  int best = 0;
  for (int r = 0; r < N; r++)
    for (int m = 0; m < sh[r]->vocab_s; m++) logits[sh[r]->vocab_off + m] = cpu_row(sh[r]->lm, m, a, 1);
  for (int m = 1; m < sh[0]->vocab; m++)
    if (logits[m] > logits[best]) best = m;
  return best;
}

// Port of dequantize_row_q4_K / dequantize_row_q6_K (llama.cpp ggml-quants.c).
static void ggml_dequant_q4_k(const uint8_t* blk, float* y) {
  const float d = h2f((uint16_t)(blk[0] | blk[1] << 8)), min = h2f((uint16_t)(blk[2] | blk[3] << 8));
  int sc[8], mn[8];
  q4_scales(blk + 4, sc, mn);
  const uint8_t* q = blk + 16;
  for (int j = 0; j < 4; j++) {
    for (int l = 0; l < 32; ++l) *y++ = d * sc[2 * j] * (q[l] & 0xF) - min * mn[2 * j];
    for (int l = 0; l < 32; ++l) *y++ = d * sc[2 * j + 1] * (q[l] >> 4) - min * mn[2 * j + 1];
    q += 32;
  }
}

static void ggml_dequant_q6_k(const uint8_t* blk, float* y) {
  const float d = h2f((uint16_t)(blk[208] | blk[209] << 8));
  const uint8_t* ql = blk;
  const uint8_t* qh = blk + 128;
  const int8_t* sc = (const int8_t*)(blk + 192);
  for (int n = 0; n < 256; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

static void dequant_row(Kind kind, const uint8_t* row, int K, std::vector<double>& out) {
  out.resize(K);
  float w[256];
  for (int sb = 0; sb < K / 256; sb++) {
    const uint8_t* blk = row + (size_t)sb * block_bytes(kind);
    kind == kQ4 ? ggml_dequant_q4_k(blk, w) : ggml_dequant_q6_k(blk, w);
    for (int e = 0; e < 256; e++) out[sb * 256 + e] = w[e];
  }
}

// The shards' slices add up to the whole model, in exact arithmetic: rows of Q and gate, columns of
// O and of the padded down (against the unsliced rows, in double).
static bool check_slices(const FullModel& fm, const std::vector<Shard*>& sh) {
  Rng rng{4242};
  bool ok = true;
  for (int layer = 0; layer < fm.n_layers; layer++) {
    const FullLayer& F = fm.layers[layer];
    struct Case {
      const char* name;
      const TView* full;
      int cols;  // input width
    };
    for (const Case& c : {Case{"o", &F.o, kHeads * kDim}, Case{"down", &F.down, kFFE}}) {
      std::vector<double> xin(c.cols);
      for (auto& v : xin) v = rng.uniform(-1.f, 1.f);
      double worst = 0;
      for (int m : {0, 1, 31, 777 % c.full->M, c.full->M - 1}) {
        std::vector<double> w;
        dequant_row(c.full->kind, c.full->data + (size_t)m * c.full->row_bytes(), c.cols, w);
        double ref = 0, got = 0;
        for (int k = 0; k < c.cols; k++) ref += w[k] * xin[k];
        for (Shard* s : sh) {
          const Tensor& t = c.full == &F.o ? s->layers[layer].o : s->layers[layer].down;
          int g0, ng, e;
          layer_slices(s->rank, layer, g0, ng, e);
          const int base = c.full == &F.o ? 8 * g0 * kDim : down_first_sb(e) * 256;
          dequant_row(t.kind, t.packed.data() + (size_t)m * (t.K / 256) * block_bytes(t.kind), t.K, w);
          for (int k = 0; k < t.K; k++) {
            const int gk = base + k;
            const bool mine = c.full == &F.o ? k < 8 * ng * kDim : gk >= e * kFFS && gk < (e + 1) * kFFS;
            if (mine) got += w[k] * xin[gk];
            else if (w[k] != 0) ok = false;  // padding must dequantize to zero
          }
        }
        worst = std::max(worst, fabs(got - ref) / (fabs(ref) + 1e-9));
      }
      printf("{\"check\":\"slice_sum\",\"layer\":%d,\"tensor\":\"%s\",\"kind\":\"%s\",\"max_rel\":%.2e}\n", layer,
             c.name, c.full->kind == kQ4 ? "q4_K" : "q6_K", worst);
      if (worst > 1e-9) ok = false;
    }
  }
  return ok;
}

static void fill_kv_random(Shard& sh, CpuKV* cs, Rng& rng) {
  const size_t n = (size_t)sh.n_layers * sh.s.max_ctx * kKVS * kDim;
  std::vector<uint16_t> k(n), v(n);
  for (auto& h : k) h = f2h(rng.uniform(-1.f, 1.f));
  for (auto& h : v) h = f2h(rng.uniform(-1.f, 1.f));
  CHECK(hipSetDevice(sh.device));
  CHECK(hipMemcpy(sh.s.kbase, k.data(), n * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(sh.s.vbase, v.data(), n * 2, hipMemcpyHostToDevice));
  if (cs) {
    cs->max_ctx = sh.s.max_ctx;
    cs->kc.swap(k);
    cs->vc.swap(v);
  }
}

static int device_count() {
  int n = 0;
  CHECK(hipGetDeviceCount(&n));
  return n;
}

static void set_token(Shard& sh, int token) {
  CHECK(hipSetDevice(sh.device));
  CHECK(hipMemcpy(sh.s.token, &token, 4, hipMemcpyHostToDevice));
}

static std::vector<float> gather_logits(const std::vector<Shard*>& sh) {
  std::vector<float> all(sh[0]->vocab);
  for (Shard* s : sh) {
    CHECK(hipSetDevice(s->device));
    CHECK(hipMemcpy(all.data() + s->vocab_off, s->s.logits, s->vocab_s * 4, hipMemcpyDeviceToHost));
  }
  return all;
}

static bool compare_logits(const char* what, int pos, const float* got, const float* ref, int vocab, int tok,
                           int ref_tok) {
  const Err e = max_err(got, ref, vocab);
  printf("{\"check\":\"%s\",\"pos\":%d,\"max_rel\":%.2e,\"ref_max\":%.3g,\"nonfinite\":%d,\"argmax\":%d,"
         "\"argmax_ref\":%d}\n",
         what, pos, e.rel, e.ref_max, e.nonfinite, tok, ref_tok);
  return !(e.rel > 1e-3 || e.nonfinite || e.ref_max == 0 || tok != ref_tok);
}

static int run_check() {
  const int n_layers = 2, vocab = 512, max_ctx = 2304;  // past kSeqTokens: both attention layouts
  Rng rng{0x9E3779B97F4A7C15ull};
  FullModel fm;
  fprintf(stderr, "%s, %d shards: building a random %d-layer model\n", kModelName, kShards, n_layers);
  random_model(fm, n_layers, vocab, rng, 1e-4f, 3e-4f);
  const int ndev = device_count();
  std::vector<std::unique_ptr<Shard>> own;
  std::vector<Shard*> sh;
  for (int r = 0; r < kShards; r++) {
    own.emplace_back(new Shard());
    build_shard(*own.back(), fm, r, r % ndev, true);
    alloc_state(*own.back(), max_ctx, true);
    sh.push_back(own.back().get());
  }
  Mesh mesh(kShards);
  Run run;
  run.shards = sh;
  for (int r = 0; r < kShards; r++) run.tps.push_back(new MeshTransport(&mesh, r));
  std::vector<CpuKV> kv(kShards);
  for (int r = 0; r < kShards; r++) fill_kv_random(*sh[r], &kv[r], rng);

  bool ok = check_slices(fm, sh);
  int token = 11;
  std::vector<float> ref_trace, ref_logits;
  std::vector<int> ref_sel;
  for (int pos : {0, 37, 300, 2100}) {
    for (Shard* s : sh) set_token(*s, token);
    launch_all(run, pos, 1, false);
    const int ref_tok = cpu_decode(sh, kv, token, pos, ref_trace, ref_logits, ref_sel);
    std::vector<std::vector<float>> traces(kShards, std::vector<float>(ref_trace.size()));
    std::vector<int> sel(ref_sel.size());
    int tok;
    for (int r = 0; r < kShards; r++) {
      CHECK(hipSetDevice(sh[r]->device));
      CHECK(hipMemcpy(traces[r].data(), sh[r]->s.trace, ref_trace.size() * 4, hipMemcpyDeviceToHost));
    }
    CHECK(hipSetDevice(sh[0]->device));
    CHECK(hipMemcpy(sel.data(), sh[0]->s.trace_sel, sel.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&tok, sh[0]->s.token, 4, hipMemcpyDeviceToHost));
    const std::vector<float> logits = gather_logits(sh);
    int same = 1;
    for (int r = 1; r < kShards; r++)
      same &= memcmp(traces[r].data(), traces[0].data(), traces[0].size() * 4) == 0;
    printf("{\"check\":\"x_identical_on_every_shard\",\"pos\":%d,\"ok\":%s}\n", pos, same ? "true" : "false");
    ok &= same != 0;
    int kv_bad = 0;
    for (int r = 0; r < kShards; r++) {
      std::vector<uint16_t> kc(kv[r].kc.size());
      CHECK(hipSetDevice(sh[r]->device));
      CHECK(hipMemcpy(kc.data(), sh[r]->s.kbase, kc.size() * 2, hipMemcpyDeviceToHost));
      for (int layer = 0; layer < n_layers; layer++)
        for (int i = 0; i < kv_stride(layer_role(sh[r]->rank, layer)) * kDim; i++) {
          const size_t o = kv[r].off(layer, pos, 0, kv_stride(layer_role(sh[r]->rank, layer))) + i;
          if (fabsf(h2f(kc[o]) - h2f(kv[r].kc[o])) > 1e-2f * (1.f + fabsf(h2f(kv[r].kc[o])))) kv_bad++;
        }
    }
    for (int l = 0; l <= n_layers; l++) {
      const Err e = max_err(&traces[0][(size_t)l * kHidden], &ref_trace[(size_t)l * kHidden], kHidden);
      printf("{\"check\":\"hidden\",\"pos\":%d,\"after_layer\":%d,\"max_rel\":%.2e,\"ref_max\":%.3g,\"nonfinite\":%d}\n",
             pos, l - 1, e.rel, e.ref_max, e.nonfinite);
      if (e.rel > 1e-3 || e.nonfinite || e.ref_max == 0) ok = false;
    }
    int sel_bad = 0;
    for (size_t i = 0; i < sel.size(); i++) sel_bad += sel[i] != ref_sel[i];
    printf("{\"check\":\"experts\",\"pos\":%d,\"mismatch\":%d}\n", pos, sel_bad);
    if (sel_bad) ok = false;
    ok &= compare_logits("logits", pos, logits.data(), ref_logits.data(), vocab, tok, ref_tok);
    if (kv_bad) {
      printf("{\"check\":\"kv\",\"pos\":%d,\"mismatch\":%d}\n", pos, kv_bad);
      ok = false;
    }
    token = ref_tok;
  }

  // Several tokens in one launch: each step's logits bit-identical to the same steps as one-step launches
  // from the same caches, and the tokens the CPU's. Against the CPU the logits only to 1e-2: once a K or V
  // value rounds to the other fp16 neighbour on one side, the steps after it drift apart (expf differs).
  const int pos0 = 301, steps = 5;
  std::vector<std::vector<uint16_t>> saved(2 * kShards);
  const size_t kv_bytes = (size_t)n_layers * max_ctx * kKVS * kDim * 2;
  for (int r = 0; r < kShards; r++) {
    CHECK(hipSetDevice(sh[r]->device));
    for (int c = 0; c < 2; c++) {
      saved[2 * r + c].resize(kv_bytes / 2);
      CHECK(hipMemcpy(saved[2 * r + c].data(), c ? sh[r]->s.vbase : sh[r]->s.kbase, kv_bytes, hipMemcpyDeviceToHost));
    }
  }
  std::vector<float> single((size_t)steps * vocab);
  for (Shard* s : sh) set_token(*s, token);
  for (int i = 0; i < steps; i++) {
    launch_all(run, pos0 + i, 1, false);
    const std::vector<float> l = gather_logits(sh);
    memcpy(&single[(size_t)i * vocab], l.data(), vocab * 4);
  }
  for (int r = 0; r < kShards; r++) {
    CHECK(hipSetDevice(sh[r]->device));
    for (int c = 0; c < 2; c++)
      CHECK(hipMemcpy(c ? sh[r]->s.vbase : sh[r]->s.kbase, saved[2 * r + c].data(), kv_bytes, hipMemcpyHostToDevice));
  }
  for (Shard* s : sh) set_token(*s, token);
  std::vector<float*> d_lg;
  for (Shard* s : sh) {
    CHECK(hipSetDevice(s->device));
    float* p;
    CHECK(hipMalloc(&p, (size_t)steps * s->vocab_s * 4));
    d_lg.push_back(p);
  }
  launch_all(run, pos0, steps, false, {}, d_lg);
  std::vector<int> toks(steps);
  CHECK(hipSetDevice(sh[0]->device));
  CHECK(hipMemcpy(toks.data(), sh[0]->s.tokens_out, steps * 4, hipMemcpyDeviceToHost));
  std::vector<float> all((size_t)steps * vocab);
  for (int r = 0; r < kShards; r++) {
    std::vector<float> part((size_t)steps * sh[r]->vocab_s);
    CHECK(hipSetDevice(sh[r]->device));
    CHECK(hipMemcpy(part.data(), d_lg[r], part.size() * 4, hipMemcpyDeviceToHost));
    for (int i = 0; i < steps; i++)
      memcpy(&all[(size_t)i * vocab + sh[r]->vocab_off], &part[(size_t)i * sh[r]->vocab_s], sh[r]->vocab_s * 4);
  }
  const bool same = memcmp(all.data(), single.data(), all.size() * 4) == 0;
  printf("{\"check\":\"multi_step_vs_single_steps\",\"steps\":%d,\"bit_identical\":%s}\n", steps,
         same ? "true" : "false");
  ok &= same;
  for (int i = 0; i < steps; i++) {
    const int ref_tok = cpu_decode(sh, kv, token, pos0 + i, ref_trace, ref_logits, ref_sel);
    const Err e = max_err(&all[(size_t)i * vocab], ref_logits.data(), vocab);
    printf("{\"check\":\"multi_step\",\"pos\":%d,\"max_rel\":%.2e,\"token\":%d,\"token_ref\":%d}\n", pos0 + i,
           e.rel, toks[i], ref_tok);
    ok &= toks[i] == ref_tok && e.rel < 1e-2 && !e.nonfinite;
    token = ref_tok;
  }
  fprintf(stderr, ok ? "check passed\n" : "check FAILED\n");
  return ok ? 0 : 1;
}

// ------------------------------------------------------------ real weights
static void load_shards(FullModel& fm, const char* gguf, int layers, const std::vector<int>& ranks,
                        std::vector<std::unique_ptr<Shard>>& own, std::vector<Shard*>& sh, int max_ctx) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string path = gguf;
  const bool slice = path.size() > 8 && path.compare(path.size() - 8, 8, ".mkslice") == 0;
  if (slice && ranks.size() != 1) {
    fprintf(stderr, "a slice file holds one rank\n");
    exit(1);
  }
  if (!slice) model_from_gguf(fm, gguf, layers);
  const int ndev = device_count();
  for (int r : ranks) {
    own.emplace_back(new Shard());
    if (slice) load_slice(*own.back(), gguf, r, (int)sh.size() % ndev, layers);
    else build_shard(*own.back(), fm, r, (int)sh.size() % ndev, false);
    alloc_state(*own.back(), max_ctx, false);
    sh.push_back(own.back().get());
    fprintf(stderr, "rank %d of %d on GPU %d: %.2f GB of weights, %.3f GB read per token (%d layers)\n", r, kShards,
            sh.back()->device, sh.back()->weight_bytes / 1e9, sh.back()->token_bytes / 1e9, sh.back()->n_layers);
  }
  fprintf(stderr, "loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

static int run_bench(const char* gguf, int layers, double gbps_peak) {
  const bool solo = kShards > 1 && getenv("MK_SOLO") && atoi(getenv("MK_SOLO"));
  const int max_ctx = 4096 + 256;
  FullModel fm;
  std::vector<std::unique_ptr<Shard>> own;
  std::vector<Shard*> sh;
  std::vector<int> ranks;
  for (int r = 0; r < (solo ? 1 : kShards); r++) ranks.push_back(r);
  load_shards(fm, gguf, layers, ranks, own, sh, max_ctx);
  Mesh mesh(kShards);
  Run run;
  run.shards = sh;
  for (int r = 0; r < (int)sh.size(); r++)
    run.tps.push_back(solo ? (Transport*)new LocalTransport() : new MeshTransport(&mesh, r));
  Rng rng{12345};
  for (Shard* s : sh) {
    fill_kv_random(*s, nullptr, rng);
    set_token(*s, 1);
  }
  int wall_khz;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, sh[0]->device));
  const int nl = sh[0]->n_layers;
  const int per_layer = kShards > 1 ? 7 : 5;
  const char* names7[7] = {"qkv", "attn", "o_xchg", "xup_router", "gate_up", "down_xchg", "xup"};
  const char* names5[5] = {"qkv", "attn", "o", "gate_up", "down"};
  for (int depth : {0, 1024, 4096}) {
    launch_all(run, depth, 16, false);
    const auto t0 = std::chrono::steady_clock::now();
    launch_all(run, depth, 128, false);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 128;
    launch_all(run, depth + 64, 1, true);
    std::vector<unsigned long long> st(kMaxStamps);
    CHECK(hipSetDevice(sh[0]->device));
    CHECK(hipMemcpy(st.data(), sh[0]->s.stamps, st.size() * 8, hipMemcpyDeviceToHost));
    auto dt = [&](int i) { return (double)(st[i + 1] - st[i]) * 1e3 / wall_khz; };
    const int stamped = std::min(nl, (kMaxStamps - 4) / per_layer);
    double stage[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int l = 0; l < stamped; l++)
      for (int k = 0; k < per_layer; k++) stage[k] += dt(1 + per_layer * l + k) * nl / stamped;
    const double kv_bytes = (double)nl * (depth + 64) * kKVS * kDim * 2 * 2;
    const double floor_us = (sh[0]->token_bytes + kv_bytes) / (gbps_peak * 1e3);
    printf("{\"kernel\":\"qwen3moe_shard\",\"model\":\"%s\",\"shards\":%d,\"solo\":%s,\"layers\":%d,\"grid\":%d,"
           "\"depth\":%d,\"tok_per_s\":%.2f,\"us_per_token\":%.1f,\"us_per_layer\":%.2f,\"token_mb\":%.1f,"
           "\"kv_mb\":%.1f,\"gbps\":%.0f,\"floor_us\":%.1f,\"of_floor\":%.3f,\"stage_us\":{",
           kModelName, kShards, solo ? "true" : "false", nl, kGrid, depth, 1e6 / us, us, us / nl,
           sh[0]->token_bytes / 1e6, kv_bytes / 1e6, (sh[0]->token_bytes + kv_bytes) / (us * 1e3), floor_us,
           floor_us / us);
    for (int k = 0; k < per_layer; k++)
      printf("%s\"%s\":%.1f", k ? "," : "", (kShards > 1 ? names7 : names5)[k], stage[k]);
    printf("}}\n");
    fflush(stdout);
  }
  return 0;
}

// Teacher-forced against tools/llama_ref.cpp output, then free running; all shards in this process.
static int run_compare(const char* gguf, const std::string& ref, int layers) {
  FullModel fm;
  std::vector<std::unique_ptr<Shard>> own;
  std::vector<Shard*> sh;
  std::vector<int> ranks;
  for (int r = 0; r < kShards; r++) ranks.push_back(r);
  load_shards(fm, gguf, layers, ranks, own, sh, 4096);
  Reference R;
  if (!read_reference(ref, fm.vocab, R)) {
    fprintf(stderr, "cannot read %s.txt / .logits\n", ref.c_str());
    return 1;
  }
  Mesh mesh(kShards);
  Run run;
  run.shards = sh;
  for (int r = 0; r < kShards; r++) run.tps.push_back(new MeshTransport(&mesh, r));
  const int P = (int)R.prompt.size(), G = (int)R.gen.size(), V = fm.vocab;
  std::vector<int> inputs(R.prompt);
  inputs.insert(inputs.end(), R.gen.begin(), R.gen.end() - 1);
  const int steps = (int)inputs.size();
  std::vector<int*> d_in;
  std::vector<float*> d_lg;
  for (Shard* s : sh) {
    CHECK(hipSetDevice(s->device));
    int* di;
    float* dl;
    CHECK(hipMalloc(&di, steps * 4));
    CHECK(hipMalloc(&dl, (size_t)steps * s->vocab_s * 4));
    CHECK(hipMemcpy(di, inputs.data(), steps * 4, hipMemcpyHostToDevice));
    d_in.push_back(di);
    d_lg.push_back(dl);
  }
  launch_all(run, 0, steps, false, d_in, d_lg);
  std::vector<float> ours((size_t)G * V);
  for (size_t i = 0; i < sh.size(); i++) {
    std::vector<float> part((size_t)steps * sh[i]->vocab_s);
    CHECK(hipSetDevice(sh[i]->device));
    CHECK(hipMemcpy(part.data(), d_lg[i], part.size() * 4, hipMemcpyDeviceToHost));
    for (int g = 0; g < G; g++)
      memcpy(&ours[(size_t)g * V + sh[i]->vocab_off], &part[(size_t)(P - 1 + g) * sh[i]->vocab_s],
             sh[i]->vocab_s * 4);
  }
  if (FILE* f = fopen((ref + ".shard.logits").c_str(), "wb")) {
    fwrite(ours.data(), 4, ours.size(), f);
    fclose(f);
  }
  int agree = 0;
  print_logit_comparison(ours.data(), R, V, agree);
  return agree == G ? 0 : 1;
}

// One shard per process (one per box), exchanges over TCP: decode `tokens` greedy tokens after warmup.
static int run_one(const char* gguf, int rank, const std::string& peers, int layers, int tokens) {
  Transport* tp = open_transport(rank, peers);
  if (tp->size != kShards) {
    fprintf(stderr, "this build has %d shards, %d peers given\n", kShards, tp->size);
    return 1;
  }
  FullModel fm;
  std::vector<std::unique_ptr<Shard>> own;
  std::vector<Shard*> sh;
  load_shards(fm, gguf, layers, {rank}, own, sh, 4096 + 256);
  Run run;
  run.shards = sh;
  run.tps.push_back(tp);
  Rng rng{12345};
  fill_kv_random(*sh[0], nullptr, rng);
  set_token(*sh[0], 1);
  tp->barrier();
  for (int depth : {0, 4096 - tokens}) {
    launch_all(run, depth, 16, false);
    tp->barrier();
    const auto t0 = std::chrono::steady_clock::now();
    launch_all(run, depth, tokens, false);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / tokens;
    std::vector<int> toks(std::min(tokens, 8));
    CHECK(hipMemcpy(toks.data(), sh[0]->s.tokens_out, toks.size() * 4, hipMemcpyDeviceToHost));
    if (rank == 0) {
      printf("{\"kernel\":\"qwen3moe_shard\",\"model\":\"%s\",\"shards\":%d,\"transport\":\"%s\",\"layers\":%d,"
             "\"depth\":%d,\"tokens\":%d,\"tok_per_s\":%.2f,\"us_per_token\":%.1f,\"token_mb_per_box\":%.1f,\"first\":[",
             kModelName, kShards, tp->name(), sh[0]->n_layers, depth, tokens, 1e6 / us, us, sh[0]->token_bytes / 1e6);
      for (size_t i = 0; i < toks.size(); i++) printf("%s%d", i ? "," : "", toks[i]);
      printf("]}\n");
    }
    fflush(stdout);
  }
  return 0;
}

// FNV-1a of a slice's bytes: llama.cpp's slices (library, MK_TP_SUMS=1) against the GGUF's (`slicesums`).
static uint64_t slice_sum(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
  return h;
}

#ifdef MK_LAYOUT_LLAMA
// The slices llama.cpp gives rank `rank` (before padding), from the GGUF: what MK_TP_SUMS prints in llama.cpp.
static int run_slicesums(const char* gguf, int rank, int layers) {
  FullModel fm;
  model_from_gguf(fm, gguf, layers);
  auto pr = [&](int l, const char* name, const void* p, size_t n) {
    printf("sum rank %d layer %d %s %016llx %zu\n", rank, l, name, (unsigned long long)slice_sum(p, n), n);
  };
  for (int l = 0; l < fm.n_layers; l++) {
    const FullLayer& F = fm.layers[l];
    int g0, ng, e;
    layer_slices(rank, l, g0, ng, e);
    const int h0 = 8 * g0, nh = 8 * ng, kv0 = g0 / kIG, nkv = kv_count(layer_role(rank, l));
    const Tensor q = rows_of(F.q, h0 * kDim, nh * kDim), k = rows_of(F.k, kv0 * kDim, nkv * kDim);
    const Tensor v = rows_of(F.v, kv0 * kDim, nkv * kDim), o = cols_of(F.o, h0 * kDim / 256, nh * kDim / 256);
    const Tensor g = expert_rows(F.gate, kFFE, e * kFFS, kFFS), u = expert_rows(F.up, kFFE, e * kFFS, kFFS);
    const Tensor d = cols_of(F.down, e * kFFS / 256, kFFS / 256);
    pr(l, "attn_q", q.packed.data(), q.packed.size());
    pr(l, "attn_k", k.packed.data(), k.packed.size());
    pr(l, "attn_v", v.packed.data(), v.packed.size());
    pr(l, "attn_output", o.packed.data(), o.packed.size());
    pr(l, "ffn_gate_exps", g.packed.data(), g.packed.size());
    pr(l, "ffn_up_exps", u.packed.data(), u.packed.size());
    pr(l, "ffn_down_exps", d.packed.data(), d.packed.size());
    pr(l, "ffn_gate_inp", F.router, (size_t)kExperts * kHidden * 4);
    pr(l, "attn_norm", F.attn_norm, kHidden * 4);
    pr(l, "ffn_norm", F.ffn_norm, kHidden * 4);
    pr(l, "attn_q_norm", F.q_norm, kDim * 4);
    pr(l, "attn_k_norm", F.k_norm, kDim * 4);
  }
  // LM head: llama.cpp's rows for rank (sizes rotated by n_layer % 3, slices in device order)
  const int V = fm.vocab, rot = fm.n_layers % kShards;
  auto size = [&](int d) {
    const int j = (d - rot + kShards) % kShards;
    return (j == kShards - 1 ? V : V * (j + 1) / kShards) - V * j / kShards;
  };
  int lo = 0;
  for (int d = 0; d < rank; d++) lo += size(d);
  const int hi = lo + size(rank);
  const Tensor lm = rows_of(fm.lm, lo, hi - lo);
  printf("sum rank %d output %016llx %zu rows %d off %d\n", rank, (unsigned long long)slice_sum(lm.packed.data(), lm.packed.size()),
         lm.packed.size(), hi - lo, lo);
  printf("sum rank %d output_norm %016llx\n", rank, (unsigned long long)slice_sum(fm.out_norm, kHidden * 4));
  return 0;
}
#endif

#ifdef MK_LIBRARY
// ------------------------------------------------------- llama.cpp library
// One device of llama.cpp's tensor split: its slices of llama.cpp's weights are copied to the host, padded to
// this layout and re-tiled (build_shard's steps after the slicing), and its KV cache is read in place.
struct MkTp {
  Shard sh;
  Run run;
  std::vector<std::pair<void*, bool>> allocs;
  Transport* tp = nullptr;
  int* d_in = nullptr;
  float* d_lg = nullptr;
  int rows = 0;           // LM head rows of this device
  bool emb_row = false;   // the embedding comes one row per token
};

constexpr int32_t kGgmlF32 = 0, kGgmlQ4K = 12, kGgmlQ6K = 14;

static int lib_create(const mk_tp_model& m, MkTp& mk, std::string& why) {
  if (m.n_ranks != kShards || m.rank < 0 || m.rank >= kShards) return why = "this build is for 3 devices", 1;
  int dev = 0, per_cu = 0, cus = 0;
  CHECK(hipGetDevice(&dev));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, decode, kBlock, 0));
  CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));
  if (kGrid > per_cu * cus) return why = "the grid's blocks cannot all be resident", 1;

  std::string bad;
  static const bool sums = getenv("MK_TP_SUMS") && atoi(getenv("MK_TP_SUMS"));
  int cur_layer = -1;
  auto host = [&](const mk_tp_tensor& t, const char* name, int32_t type, int64_t ne0, int64_t ne1, int64_t ne2,
                  std::vector<uint8_t>& out) {
    if (!bad.empty()) return;
    if (t.data == nullptr || t.type != type || t.ne[0] != ne0 || t.ne[1] != ne1 || t.ne[2] != ne2) {
      char b[160];
      snprintf(b, sizeof(b), "%s: type %d [%lld %lld %lld], expected type %d [%lld %lld %lld]", name, t.type,
               (long long)t.ne[0], (long long)t.ne[1], (long long)t.ne[2], type, (long long)ne0, (long long)ne1,
               (long long)ne2);
      bad = b;
      return;
    }
    const size_t row = type == kGgmlF32 ? ne0 * 4 : (size_t)(ne0 / 256) * (type == kGgmlQ4K ? kQ4Bytes : kQ6Bytes);
    out.resize(row * ne1 * ne2);
    CHECK(hipMemcpy(out.data(), t.data, out.size(), hipMemcpyDeviceToHost));
    if (sums) fprintf(stderr, "sum rank %d layer %d %s %016llx %zu\n", m.rank, cur_layer, name,
                      (unsigned long long)slice_sum(out.data(), out.size()), out.size());
  };
  auto quant = [&](const mk_tp_tensor& t, const char* name, bool q6_ok, int64_t K, int64_t M, int64_t E = 1) {
    Tensor r{t.type == kGgmlQ6K ? kQ6 : kQ4, (int)(M * E), (int)K};
    host(t, name, q6_ok && t.type == kGgmlQ6K ? kGgmlQ6K : kGgmlQ4K, K, M, E, r.packed);
    return r;
  };
  auto f32 = [&](const mk_tp_tensor& t, const char* name, int64_t n0, int64_t n1 = 1) {
    std::vector<uint8_t> b;
    host(t, name, kGgmlF32, n0, n1, 1, b);
    std::vector<float> v(b.size() / 4);
    memcpy(v.data(), b.data(), b.size());
    return v;
  };

  Shard& sh = mk.sh;
  const int vocab = (int)m.tok_embd.ne[1];
  shard_meta(sh, m.rank, dev, m.n_layers, vocab);
  std::vector<Layer> dl;
  std::vector<uint16_t*> kp, vp;
  for (int l = 0; l < m.n_layers && bad.empty(); l++) {
    const mk_tp_layer& T = m.layers[l];
    cur_layer = l;
    const int r = layer_role(m.rank, l), nh = 8 * grp_count(r), nkv = kv_count(r);
    HostLayer& L = sh.layers[l];
    L.q = pad_rows(quant(T.q, "attn_q", false, kHidden, nh * kDim), kQS);
    L.k = pad_rows(quant(T.k, "attn_k", false, kHidden, nkv * kDim), kKVR);
    L.v = pad_rows(quant(T.v, "attn_v", true, kHidden, nkv * kDim), kKVR);
    L.o = pad_cols(quant(T.o, "attn_output", false, nh * kDim, kHidden), kQSB);
    L.gate = quant(T.gate, "ffn_gate_exps", false, kHidden, kFFS, kExperts);
    L.up = quant(T.up, "ffn_up_exps", false, kHidden, kFFS, kExperts);
    L.down = quant(T.down, "ffn_down_exps", true, kFFS, kHidden, kExperts);
    L.router = f32(T.router, "ffn_gate_inp", kHidden, kExperts);
    L.attn_norm = f32(T.attn_norm, "attn_norm", kHidden);
    L.ffn_norm = f32(T.ffn_norm, "ffn_norm", kHidden);
    L.q_norm = f32(T.q_norm, "attn_q_norm", kDim);
    L.k_norm = f32(T.k_norm, "attn_k_norm", kDim);
    if (!bad.empty()) break;
    upload_layer(sh, L, dl, false);
    kp.push_back((uint16_t*)T.k_cache);
    vp.push_back((uint16_t*)T.v_cache);
  }
  if (bad.empty()) {
    if (m.tok_embd.data) {
      Tensor emb = quant(m.tok_embd, "token_embd", false, kHidden, vocab);
      sh.emb.swap(emb.packed);
    } else if (m.tok_embd.type != kGgmlQ4K || m.tok_embd.ne[0] != kHidden) {
      bad = "token_embd must be Q4_K";
    } else {
      sh.emb.assign((size_t)kHSB * kQ4Bytes, 0);  // one row, written per token
      mk.emb_row = true;
    }
    mk.rows = (int)m.output.ne[1];
    Tensor lm{kQ6, mk.rows, kHidden};
    host(m.output, "output", kGgmlQ6K, kHidden, mk.rows, 1, lm.packed);
    sh.lm = pad_rows(std::move(lm), (mk.rows + 31) / 32 * 32);
    sh.out_norm = f32(m.out_norm, "output_norm", kHidden);
  }
  if (!bad.empty()) return why = bad, 1;
  sh.vocab_off = m.vocab_off;
  sh.vocab_s = sh.lm.M;
  finish_shard(sh, dl, false);
  alloc_state(sh, m.n_cells, false, &kp, &vp);
  mk.tp = open_transport(m.rank, m.peers ? m.peers : "");
  if (mk.tp->size != kShards) return why = "the peer list does not have 3 entries", 1;
  mk.run.shards = {&sh};
  mk.run.tps = {mk.tp};
  CHECK(hipMalloc(&mk.d_in, 4));
  CHECK(hipMalloc(&mk.d_lg, (size_t)sh.vocab_s * 4));
  return 0;
}
#endif

}  // namespace mk_shard

using namespace mk_shard;

#ifdef MK_LIBRARY
static void lib_free(MkTp* mk) {
  delete mk->tp;
  if (mk->sh.s.stream) (void)hipStreamDestroy(mk->sh.s.stream);
  for (auto [p, host] : mk->allocs) (void)(host ? hipHostFree(p) : hipFree(p));
  delete mk;
}

extern "C" void* mk_tp_create(const mk_tp_model* m, char* err, size_t err_len) {
  auto* mk = new MkTp();
  std::string why;
  g_allocs = &mk->allocs;
  const int rc = lib_create(*m, *mk, why);
  g_allocs = nullptr;
  if (rc != 0) {
    if (err && err_len) snprintf(err, err_len, "%s", why.c_str());
    lib_free(mk);
    return nullptr;
  }
  fprintf(stderr, "mk_tp: rank %d of %d, %d layers, %.2f GB of weights, %.3f GB read per token, %s exchanges\n",
          m->rank, kShards, m->n_layers, mk->sh.weight_bytes / 1e9, mk->sh.token_bytes / 1e9, mk->tp->name());
  fprintf(stderr, "MK_STAGE_SPLIT mode=%d rank=%d layers=%d launches=%d exchanges=%d copy=%d\n",
          MK_STAGE_SPLIT, m->rank, m->n_layers, MK_STAGE_SPLIT ? 2 * m->n_layers + 1 : 1,
          exchanges_per_step(m->n_layers), MK_STAGE_SPLIT);
  return mk;
}

extern "C" int mk_tp_decode(void* p, int32_t token, int32_t pos, const void* emb_row, float* logits) {
  auto* mk = (MkTp*)p;
  CHECK(hipSetDevice(mk->sh.device));
  if (mk->emb_row) {
    if (!emb_row) return 2;
    CHECK(hipMemcpy(mk->sh.d_emb, emb_row, (size_t)kHSB * kQ4Bytes, hipMemcpyHostToDevice));
    token = 0;
  }
  CHECK(hipMemcpy(mk->d_in, &token, 4, hipMemcpyHostToDevice));
  const double t0 = now_us();
  ProxyStats st;
  const bool ok = launch_steps(mk->run, pos, 1, false, {mk->d_in}, {mk->d_lg}, &st);
  const double us = now_us() - t0;
  // MK_TP_SLOW_MS (default 100): log steps slower than this, and every failure
  static const double slow_us = 1e3 * (getenv("MK_TP_SLOW_MS") ? atof(getenv("MK_TP_SLOW_MS")) : 100.0);
  if (!ok || us > slow_us)
    fprintf(stderr, "mk_tp: rank %d, pos %d: %s in %.1f ms; slowest exchange %u of %d: %.1f ms\n", mk->sh.rank, pos,
            ok ? "decoded" : "FAILED", us / 1e3, st.max_at, exchanges_per_step(mk->sh.n_layers), st.max_us / 1e3);
  if (!ok) return 1;
  if (logits) CHECK(hipMemcpy(logits, mk->d_lg, (size_t)mk->rows * 4, hipMemcpyDeviceToHost));
  return 0;
}

extern "C" void mk_tp_free(void* p) {
  if (p) lib_free((MkTp*)p);
}
#else
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "check";
  const double peak = getenv("MK_PEAK_GBPS") ? atof(getenv("MK_PEAK_GBPS")) : 639.0;
  if (mode == "check") return run_check();
  if (mode == "bench" && argc > 2) return run_bench(argv[2], argc > 3 ? atoi(argv[3]) : 1 << 20, peak);
  if (mode == "compare" && argc > 3) return run_compare(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 1 << 20);
  if (mode == "run" && argc > 4)
    return run_one(argv[2], atoi(argv[3]), argv[4], argc > 5 ? atoi(argv[5]) : 1 << 20, argc > 6 ? atoi(argv[6]) : 64);
  if (mode == "dump" && argc > 4) return run_dump(argv[2], atoi(argv[3]), argv[4], argc > 5 ? atoi(argv[5]) : 1 << 20);
#ifdef MK_LAYOUT_LLAMA
  if (mode == "slicesums" && argc > 3) return run_slicesums(argv[2], atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 1 << 20);
#endif
  fprintf(stderr,
          "usage: %s check | bench <gguf> [layers] | compare <gguf> <ref_prefix> [layers] |"
          " run <gguf or .mkslice> <rank> <host:port,...> [layers] [tokens] | dump <gguf> <rank> <out.mkslice or -> [layers]\n",
          argv[0]);
  return 2;
}
#endif
