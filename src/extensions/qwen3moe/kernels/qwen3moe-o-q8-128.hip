// Native arithmetic from the measured O fixture, kernel source SHA256
// 02127fcbb54cf4dc32407c1ac3d7f95a460560a9e9775ab4f6b5914d35a68bea.
// Production changes are token-sticky status and bounded finite error output.
#include "qwen3moe-o-q8-128.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace {
constexpr int hidden = 4096, selected = 8, wave_size = 32;
// Exact native stream/quantizer slice, SHA256
// 0a4b1ee2792976f3c20eb1c49a47e5ba3e705eee4502531363921ba3b078c405.
constexpr int kQ4Chunks = 9, kQ6Chunks = 13;
typedef unsigned u4v __attribute__((ext_vector_type(4)));
typedef uint16_t u8h __attribute__((ext_vector_type(8), aligned(2)));
static_assert(sizeof(u8h) == 16 && alignof(u8h) == 2, "native chunks need only halfword alignment");
typedef const u8h __attribute__((address_space(1))) * global_u8h;
typedef const uint16_t __attribute__((address_space(1))) * global_u16;
__device__ __forceinline__ u4v ldg(const uint8_t * p) {
    const u8h halves = *(global_u8h) p;
    u4v words;
    __builtin_memcpy(&words, &halves, sizeof(words));
    return words;
}
__device__ __forceinline__ uint16_t ldg(const uint16_t * p) {
    return *(global_u16) p;
}
__device__ __forceinline__ float f16_to_f32(uint16_t bits) {
    __half value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}
__device__ __forceinline__ float half_lo(unsigned bits) { return f16_to_f32(uint16_t(bits)); }
__device__ __forceinline__ float half_hi(unsigned bits) { return f16_to_f32(uint16_t(bits >> 16)); }
struct Smem {
    alignas(16) int8_t xq[hidden];
    int xs16[hidden / 16];
    float xd[hidden / 128];
    float part[selected][2][wave_size];
    int ids[selected];
    float routing[selected];
    unsigned proceed;
};
__device__ __forceinline__ void load_group(const float * src, int group, float (&values)[8]) {
    const unsigned lane = threadIdx.x % wave_size;
#pragma unroll
    for (int i = 0; i < 8; ++i) values[i] = src[group * 256 + lane * 8 + i];
}
__device__ __forceinline__ void quantize_group(Smem & sm, int group, const float (&values)[8]) {
    const unsigned lane = threadIdx.x % wave_size;
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, fabsf(values[i]));
    // Sixteen lanes own one contiguous 128-value activation group.
    maximum = warp_reduce_max<wave_size / 2>(maximum);
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    const float initial = __builtin_amdgcn_rcpf(scale);
    const float reciprocal = fmaf(fmaf(-scale, initial, 1.0f), initial, initial);
    unsigned packed[2] = {};
    int sum8 = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float quotient = values[i] * reciprocal;
        const float corrected = fmaf(fmaf(-quotient, scale, values[i]), reciprocal, quotient);
        int quantized = int(rintf(corrected));
        quantized = quantized > 127 ? 127 : quantized < -127 ? -127 : quantized;
        packed[i >> 2] |= (unsigned(quantized) & 255u) << (8 * (i & 3));
        sum8 += quantized;
    }
    *reinterpret_cast<uint2 *>(sm.xq + group * 256 + lane * 8) = make_uint2(packed[0], packed[1]);
    const int sum16 = sum8 + __shfl_xor(sum8, 1);
    if ((lane & 1) == 0) sm.xs16[group * 16 + lane / 2] = sum16;
    if ((lane & 15) == 0) sm.xd[group * 2 + lane / 16] = scale;
}
__device__ __forceinline__ unsigned load_agent(const uint32_t * p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void set_error(uint32_t * p, unsigned error) {
    __hip_atomic_fetch_or(p, error, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

__host__ __device__ constexpr int blk_chunks(unsigned q6, int b) { return (q6 >> b) & 1 ? kQ6Chunks : kQ4Chunks; }
__host__ __device__ constexpr int q6_chunk(int pos) {
  constexpr int order[kQ6Chunks] = {0, 2, 8, 1, 3, 9, 4, 6, 10, 5, 7, 11, 12};
  return order[pos];
}

template <int NB, int D, unsigned Q6, bool CHAIN, int CS = 32>
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
  static constexpr int chunk_of(int k) {
    return (Q6 >> blk_of(k)) & 1 ? q6_chunk(pos_of(k)) : pos_of(k);
  }

  const uint8_t* base[NB];  // this lane's chunk 0 of each block
  int sb[NB];           // activation superblock of each block
  uint16_t d6[NB];      // Q6_K scale of each Q6 block, loaded by pre()
  const uint16_t* dptr[NB];
  u4v buf[D];

  // Every chunk index is a template argument, so the block arrays above stay in registers
  // however the compiler would have unrolled a loop over them.
  template <int... K>
  __device__ __forceinline__ void pre_k(std::integer_sequence<int, K...>) {
    ((buf[K] = ldg(base[blk_of(K)] + chunk_of(K) * CS * 16)), ...);
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
    constexpr bool q6 = (Q6 >> b) & 1;
    const int8_t* xq = sm.xq + sb[b] * 256;
    const int* xs = sm.xs16 + sb[b] * 16;
    const u4v cur = buf[k % D];
    if constexpr (k + D < N) buf[k % D] = ldg(base[blk_of(k + D)] + chunk_of(k + D) * CS * 16);
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
        int isc0 = 0, imn0 = 0, isc1 = 0, imn1 = 0;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const int sc_lo = (s0 >> (8 * t)) & 63, mn_lo = (s1 >> (8 * t)) & 63;
          const unsigned b8 = (s2 >> (8 * t)) & 0xFF;
          const int sc_hi = (b8 & 0xF) | (((s0 >> (8 * t + 6)) & 3) << 4);
          const int mn_hi = (b8 >> 4) | (((s1 >> (8 * t + 6)) & 3) << 4);
          isc0 += __mul24(sc_lo, sumi[t]);
          imn0 += __mul24(mn_lo, xs[2 * t] + xs[2 * t + 1]);
          isc1 += __mul24(sc_hi, sumi[t + 4]);
          imn1 += __mul24(mn_hi, xs[2 * (t + 4)] + xs[2 * (t + 4) + 1]);
        }
        const float d = half_lo(hdr.x), dmin = half_hi(hdr.x);
        acc = __builtin_fmaf(sm.xd[2 * sb[b]],
                            __builtin_fmaf(d, (float)isc0, -(dmin * (float)imn0)), CHAIN ? acc : 0.f);
        acc = __builtin_fmaf(sm.xd[2 * sb[b] + 1],
                            __builtin_fmaf(d, (float)isc1, -(dmin * (float)imn1)), acc);
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
        int isum0 = 0, isum1 = 0;
#pragma unroll
        for (int s = 0; s < 16; s++) {
          const int sc = (int)(int8_t)(scw[s >> 2] >> (8 * (s & 3)));
          if (s < 8) isum0 += __mul24(sc, sumi[s] - 32 * xs[s]);
          else       isum1 += __mul24(sc, sumi[s] - 32 * xs[s]);
        }
        const float d = f16_to_f32(d6[b]);
        acc = __builtin_fmaf(sm.xd[2 * sb[b]], d * (float)isum0, CHAIN ? acc : 0.f);
        acc = __builtin_fmaf(sm.xd[2 * sb[b] + 1], d * (float)isum1, acc);
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


template<int NB, int D, unsigned Q6, bool CHAIN>
__device__ __forceinline__ void native_set(Stream<NB, D, Q6, CHAIN, 1> & stream, int block,
                                          const void * matrix, uint64_t row_stride, uint64_t expert_stride,
                                          unsigned expert, unsigned row, int superblock, int activation_block) {
    const unsigned block_bytes = ((Q6 >> block) & 1) ? sizeof(block_q6_K) : sizeof(block_q4_K);
    const auto * pointer = static_cast<const uint8_t *>(matrix) + expert * expert_stride + row * row_stride + superblock * block_bytes;
    stream.base[block] = pointer;
    stream.sb[block] = activation_block;
    if ((Q6 >> block) & 1) stream.dptr[block] = reinterpret_cast<const uint16_t *>(pointer + 208);
}


template<int Columns>
constexpr int o_rows_per_block() {
    static_assert(Columns == 2048 || Columns == 4096, "supported O input width");
    return Columns == 2048 ? 64 : 256;
}

template<int Rows>
__device__ __forceinline__ void o_zero_output(const qwen3moe_o_q8_128_args & a) {
    for (int row = threadIdx.x; row < Rows; row += 256) {
        a.output[blockIdx.x * Rows + row] = 0.0f;
    }
}

template<int Columns, int Rows>
__launch_bounds__(256)
__global__ void qwen3moe_o_native128(qwen3moe_o_q8_128_args a) {
    static_assert(Rows == 32 || Rows == 64 || Rows == 128 || Rows == 256, "bounded O row tile");
    __shared__ Smem sm;
    if (threadIdx.x == 0) sm.proceed = load_agent(a.status) == 0;
    __syncthreads();
    if (!sm.proceed) {
        o_zero_output<Rows>(a);
        return;
    }
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    constexpr int blocks_per_wave = Columns / 2048;
    float values[blocks_per_wave][8];
    bool valid = true;
#pragma unroll
    for (int b = 0; b < blocks_per_wave; ++b) {
        load_group(a.input, wave + b * 8, values[b]);
#pragma unroll
        for (int i = 0; i < 8; ++i) valid = valid && isfinite(values[b][i]);
    }
    if (!__syncthreads_and(valid)) {
        if (threadIdx.x == 0) set_error(a.status, QWEN3MOE_O_Q8_128_NONFINITE);
        o_zero_output<Rows>(a);
        return;
    }
#pragma unroll
    for (int b = 0; b < blocks_per_wave; ++b) quantize_group(sm, wave + b * 8, values[b]);
    __syncthreads();
    for (int tile = 0; tile < Rows / 32; ++tile) {
        const unsigned row = blockIdx.x * Rows + tile * 32 + lane;
        Stream<blocks_per_wave, 4, 0u, false, 1> dot;
        native_set(dot, 0, a.weights, a.row_stride, 0, 0, row, wave, wave);
        if constexpr (blocks_per_wave == 2)
            native_set(dot, 1, a.weights, a.row_stride, 0, 0, row, wave + 8, wave + 8);
        dot.pre();
        float values[blocks_per_wave];
        dot.run(sm, values);
        float value = values[0];
        if constexpr (blocks_per_wave == 2) value += values[1];
        sm.part[wave][0][lane] = value;
        __syncthreads();
        if (wave == 0) {
            value = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) value += sm.part[w][0][lane];
            if (!isfinite(value)) set_error(a.status, QWEN3MOE_O_Q8_128_NONFINITE);
            a.output[row] = isfinite(value) ? value : 0.0f;
        }
        __syncthreads();
    }
}

struct o_region {
    const void * pointer;
    size_t size, alignment;
};

bool o_valid_region(o_region r) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(r.pointer);
    return p && r.size && r.alignment && p % r.alignment == 0 && r.size <= UINTPTR_MAX - p;
}

bool o_disjoint(o_region a, o_region b) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a.pointer), y = reinterpret_cast<uintptr_t>(b.pointer);
    return x + a.size <= y || y + b.size <= x;
}
}

bool qwen3moe_o_q8_128_supported(const qwen3moe_o_q8_128_args * a) {
    if (!a || (a->columns != 2048 && a->columns != 4096)) return false;
    const size_t row = size_t(a->columns / 256) * sizeof(block_q4_K);
    if (a->row_stride != row || a->weight_bytes != 4096 * row ||
        a->input_bytes != size_t(a->columns) * sizeof(float) || a->output_bytes != 4096 * sizeof(float)) return false;
    const std::array<o_region, 4> spans = {{{a->weights, a->weight_bytes, 2},
        {a->input, a->input_bytes, 4}, {a->output, a->output_bytes, 4}, {a->status, sizeof(uint32_t), 4}}};
    for (const auto & r : spans) if (!o_valid_region(r)) return false;
    for (size_t i = 0; i < spans.size(); ++i) {
        for (size_t j = 0; j < i; ++j) if (!o_disjoint(spans[i], spans[j])) return false;
    }
    return true;
}

hipError_t qwen3moe_o_q8_128_enqueue(const qwen3moe_o_q8_128_args * a, hipStream_t stream) {
    if (!qwen3moe_o_q8_128_supported(a) || !stream) return hipErrorInvalidValue;
    if (a->columns == 2048) {
        constexpr int rows = o_rows_per_block<2048>();
        qwen3moe_o_native128<2048, rows><<<4096 / rows, 256, 0, stream>>>(*a);
    } else {
        constexpr int rows = o_rows_per_block<4096>();
        qwen3moe_o_native128<4096, rows><<<4096 / rows, 256, 0, stream>>>(*a);
    }
    return hipGetLastError();
}
