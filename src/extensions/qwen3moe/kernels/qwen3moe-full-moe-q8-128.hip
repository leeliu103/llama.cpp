// Isolated approximate full-MoE candidate. Native weight chunks are streamed
// using the AMD rdna-megakernels schedule; activations use one FP32 scale/128.
#include "qwen3moe-full-moe-q8-128.h"
#include "unary.cuh"
#include "vecdotq.cuh"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace {
constexpr int hidden = 4096, intermediate = 512, experts = 128, selected = 8;
constexpr int wave_size = 32, threads = selected * wave_size;
constexpr int down_rows = QWEN3MOE_FULL_MOE_Q8_128_DOWN_ROWS;
constexpr unsigned max_barrier_polls = 1u << 20;
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

// One interphase barrier. All workgroups participate even after invalid IDs.
// The launcher resets arrivals before every kernel, including graph replays.
// A cooperative launch and occupancy preflight are the primary residency
// checks; wall-clock and iteration bounds are independent escape mechanisms.
__device__ __forceinline__ bool arrive_and_wait(const qwen3moe_full_moe_q8_128_args & a,
                                               uint64_t timeout_ticks, unsigned & proceed) {
    __threadfence();  // every producer thread publishes its H stores
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned previous = __hip_atomic_fetch_add(a.arrival_counter, 1u, __ATOMIC_ACQ_REL,
                                                         __HIP_MEMORY_SCOPE_AGENT);
        if (previous >= gridDim.x) {
            set_error(a.sticky_error, QWEN3MOE_FULL_MOE_Q8_128_BARRIER_COUNT);
        }
        const uint64_t start = static_cast<uint64_t>(wall_clock64());
        bool arrived = false;
        for (unsigned poll = 0; poll < max_barrier_polls; ++poll) {
            if (load_agent(a.sticky_error)) {
                break;
            }
            if (load_agent(a.arrival_counter) == gridDim.x) {
                arrived = true;
                break;
            }
            if ((poll & 63u) == 63u && static_cast<uint64_t>(wall_clock64()) - start >= timeout_ticks) {
                break;
            }
        }
        if (!arrived && !load_agent(a.sticky_error)) {
            set_error(a.sticky_error, QWEN3MOE_FULL_MOE_Q8_128_BARRIER_TIMEOUT);
        }
        proceed = arrived && !load_agent(a.sticky_error);
    }
    __syncthreads();
    if (!proceed) {
        return false;
    }
    // The acquire emits gfx1151 GL1/GL0 invalidation for each consumer wave.
    (void) load_agent(a.arrival_counter);
    __syncthreads();
    return true;
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

template<bool q6, int depth>
__launch_bounds__(threads)
__global__ void full_moe_q8_128(qwen3moe_full_moe_q8_128_args a, uint64_t timeout_ticks) {
    __shared__ Smem sm;
    const unsigned lane = threadIdx.x % wave_size;
    const unsigned wave = __builtin_amdgcn_readfirstlane(threadIdx.x / wave_size);
    if (threadIdx.x < selected) {
        const int expert = a.selected_experts[threadIdx.x];
        sm.ids[threadIdx.x] = expert;
        sm.routing[threadIdx.x] = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(a.routing_weights) + threadIdx.x * a.routing_stride);
        if (expert < 0 || expert >= experts) set_error(a.sticky_error, QWEN3MOE_FULL_MOE_Q8_128_INVALID_EXPERT);
    }
    __syncthreads();
    if (threadIdx.x == 0) sm.proceed = !load_agent(a.sticky_error);
    __syncthreads();
    if (sm.proceed) {
        float values[2][8];
        load_group(a.input, wave, values[0]);
        load_group(a.input, wave + selected, values[1]);
        for (unsigned task = blockIdx.x; task < selected * intermediate / wave_size; task += gridDim.x) {
            const unsigned slot = task / (intermediate / wave_size);
            const unsigned row = (task % (intermediate / wave_size)) * wave_size + lane;
            const unsigned expert = __builtin_amdgcn_readfirstlane(unsigned(sm.ids[slot]));
            Stream<4, depth, 0u, false, 1> stream;
            native_set(stream, 0, a.gate, a.gate_row_stride, a.gate_expert_stride, expert, row, wave, wave);
            native_set(stream, 1, a.up, a.up_row_stride, a.up_expert_stride, expert, row, wave, wave);
            native_set(stream, 2, a.gate, a.gate_row_stride, a.gate_expert_stride, expert, row, wave + selected, wave + selected);
            native_set(stream, 3, a.up, a.up_row_stride, a.up_expert_stride, expert, row, wave + selected, wave + selected);
            stream.pre();
            if (task == blockIdx.x) {
                quantize_group(sm, wave, values[0]);
                quantize_group(sm, wave + selected, values[1]);
                __syncthreads();
            }
            float dots[4];
            stream.run(sm, dots);
            __syncthreads();
            sm.part[wave][0][lane] = dots[0] + dots[2];
            sm.part[wave][1][lane] = dots[1] + dots[3];
            __syncthreads();
            if (wave == 0) {
                float gate = 0.0f, up = 0.0f;
#pragma unroll
                for (int w = 0; w < selected; ++w) {
                    gate += sm.part[w][0][lane];
                    up += sm.part[w][1][lane];
                }
                a.activation_scratch[slot * intermediate + row] = up * ggml_cuda_op_silu_single(gate);
            }
        }
    }
    if (!arrive_and_wait(a, timeout_ticks, sm.proceed)) return;

    float values[2][8];
    load_group(a.activation_scratch, wave * 2, values[0]);
    load_group(a.activation_scratch, wave * 2 + 1, values[1]);
    const unsigned expert = __builtin_amdgcn_readfirstlane(unsigned(sm.ids[wave]));
    for (unsigned tile = blockIdx.x; tile < hidden / wave_size; tile += gridDim.x) {
        const unsigned row = tile * wave_size + lane;
        Stream<2, depth, q6 ? 3u : 0u, true, 1> stream;
        native_set(stream, 0, a.down, a.down_row_stride, a.down_expert_stride, expert, row, 0, wave * 2);
        native_set(stream, 1, a.down, a.down_row_stride, a.down_expert_stride, expert, row, 1, wave * 2 + 1);
        stream.pre();
        if (tile == blockIdx.x) {
            quantize_group(sm, wave * 2, values[0]);
            quantize_group(sm, wave * 2 + 1, values[1]);
            __syncthreads();
        }
        float dots[2];
        stream.run(sm, dots);
        __syncthreads();
        sm.part[wave][0][lane] = dots[1];
        __syncthreads();
        if (wave == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < selected; ++e) sum = fmaf(sm.routing[e], sm.part[e][0][lane], sum);
            a.output[row] = sum;
        }
    }
}

bool matrix_span(uint64_t row, uint64_t expert, uint64_t row_bytes, int rows, uint64_t alignment,
                 uint64_t & span) {
    if (row < row_bytes || row % alignment || expert % alignment ||
        row > uint64_t(PTRDIFF_MAX) / rows) return false;
    const uint64_t matrix = (rows - 1) * row + row_bytes;
    if (expert < matrix || expert > uint64_t(PTRDIFF_MAX) / experts) return false;
    span = (experts - 1) * expert + matrix;
    return true;
}
struct memory_span { const void * pointer; uint64_t bytes; };
bool overlaps(memory_span a, memory_span b) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a.pointer), y = reinterpret_cast<uintptr_t>(b.pointer);
    return x < y + b.bytes && y < x + a.bytes;
}

template<bool q6, int depth>
hipError_t occupancy(int * active) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(active, full_moe_q8_128<q6, depth>, threads, 0);
}
template<bool q6, int depth>
hipError_t submit(const qwen3moe_full_moe_q8_128_args & a, uint64_t timeout, hipStream_t stream) {
    auto copy = a;
    void * parameters[] = {&copy, &timeout};
    return hipLaunchCooperativeKernel(reinterpret_cast<const void *>(full_moe_q8_128<q6, depth>),
                                     dim3(a.grid_blocks), dim3(threads), parameters, 0, stream);
}
} // namespace

extern "C" int qwen3moe_full_moe_q8_128_supported(const qwen3moe_full_moe_q8_128_args * a) {
    if (!a || !a->gate || !a->up || !a->down || !a->input || !a->selected_experts || !a->routing_weights ||
        !a->output || !a->activation_scratch || !a->arrival_counter || !a->sticky_error ||
        (a->down_type != GGML_TYPE_Q4_K && a->down_type != GGML_TYPE_Q6_K) ||
        (a->grid_blocks != 32 && a->grid_blocks != 64 && a->grid_blocks != 128) ||
        (a->stream_depth != 2 && a->stream_depth != 4)) return 0;
    for (const void * p : {a->gate, a->up, a->down, static_cast<const void *>(a->input),
         static_cast<const void *>(a->selected_experts), static_cast<const void *>(a->routing_weights),
         static_cast<const void *>(a->output), static_cast<const void *>(a->activation_scratch),
         static_cast<const void *>(a->arrival_counter), static_cast<const void *>(a->sticky_error)}) {
        if (reinterpret_cast<uintptr_t>(p) % alignof(uint32_t)) return 0;
    }
    uint64_t gate_bytes, up_bytes, down_bytes;
    const uint64_t down_block = a->down_type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K);
    if (!matrix_span(a->gate_row_stride, a->gate_expert_stride, 16 * sizeof(block_q4_K), intermediate, 4, gate_bytes) ||
        !matrix_span(a->up_row_stride, a->up_expert_stride, 16 * sizeof(block_q4_K), intermediate, 4, up_bytes) ||
        !matrix_span(a->down_row_stride, a->down_expert_stride, 2 * down_block, hidden, a->down_type == GGML_TYPE_Q4_K ? 4 : 2, down_bytes) ||
        a->routing_stride < sizeof(float) || a->routing_stride % sizeof(float) ||
        a->routing_stride > uint64_t(PTRDIFF_MAX) / selected) return 0;
    const memory_span reads[] = {{a->gate, gate_bytes}, {a->up, up_bytes}, {a->down, down_bytes},
        {a->input, hidden * sizeof(float)}, {a->selected_experts, selected * sizeof(int32_t)},
        {a->routing_weights, (selected - 1) * a->routing_stride + sizeof(float)}};
    const memory_span writes[] = {{a->output, hidden * sizeof(float)},
        {a->activation_scratch, selected * intermediate * sizeof(float)}, {a->arrival_counter, 4}, {a->sticky_error, 4}};
    for (const auto & r : reads) {
        if (reinterpret_cast<uintptr_t>(r.pointer) > UINTPTR_MAX - r.bytes) return 0;
    }
    for (const auto & w : writes) {
        if (reinterpret_cast<uintptr_t>(w.pointer) > UINTPTR_MAX - w.bytes) return 0;
        for (const auto & r : reads) if (overlaps(w, r)) return 0;
        for (const auto & other : writes) if (w.pointer != other.pointer && overlaps(w, other)) return 0;
    }
    // Identical writable pointers also overlap and are never permitted.
    for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) if (writes[i].pointer == writes[j].pointer) return 0;
    return 1;
}

extern "C" hipError_t qwen3moe_full_moe_q8_128_query(const qwen3moe_full_moe_q8_128_args * a, qwen3moe_full_moe_q8_128_launch_info * info) {
    if (!info || !qwen3moe_full_moe_q8_128_supported(a)) return hipErrorInvalidValue;
    *info = {};
    hipError_t error = hipGetDevice(&info->device);
    if (error != hipSuccess) return error;
    hipDeviceProp_t props{};
    error = hipGetDeviceProperties(&props, info->device);
    if (error != hipSuccess) return error;
    if (props.warpSize != 32 || std::strncmp(props.gcnArchName, "gfx1151", 7) != 0 || !props.cooperativeLaunch) return hipErrorNotSupported;
    int wall_clock_khz = 0;
    error = hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, info->device);
    if (error != hipSuccess) return error;
    if (wall_clock_khz <= 0) return hipErrorNotSupported;
    info->multiprocessors = props.multiProcessorCount;
    info->grid_blocks = a->grid_blocks;
    info->down_type = a->down_type;
    info->stream_depth = a->stream_depth;
    info->down_rows = down_rows;
    info->down_mapping = QWEN3MOE_FULL_MOE_Q8_128_DOWN_MAPPING;
    info->barrier_timeout_ticks = uint64_t(wall_clock_khz) * 10; // 10 ms; hard poll cap also applies
    if (a->down_type == GGML_TYPE_Q6_K) {
        error = a->stream_depth == 2 ? occupancy<true, 2>(&info->active_blocks_per_multiprocessor)
                                           : occupancy<true, 4>(&info->active_blocks_per_multiprocessor);
    } else {
        error = a->stream_depth == 2 ? occupancy<false, 2>(&info->active_blocks_per_multiprocessor)
                                           : occupancy<false, 4>(&info->active_blocks_per_multiprocessor);
    }
    if (error != hipSuccess) return error;
    // This specialization hit the bounded watchdog at grid128 despite a raw
    // HIP occupancy result of seven blocks/WGP. Admit only the tested grid cap.
    if (a->grid_blocks > 64 || info->active_blocks_per_multiprocessor <= 0 ||
        int64_t(info->active_blocks_per_multiprocessor) * info->multiprocessors < a->grid_blocks) return hipErrorCooperativeLaunchTooLarge;
    return hipSuccess;
}

extern "C" hipError_t qwen3moe_full_moe_q8_128_enqueue(const qwen3moe_full_moe_q8_128_args * a,
                                               const qwen3moe_full_moe_q8_128_launch_info * info, hipStream_t stream) {
    if (!info || !qwen3moe_full_moe_q8_128_supported(a) || a->grid_blocks > 64 || info->device < 0 ||
        info->grid_blocks != a->grid_blocks || info->down_type != a->down_type ||
        info->stream_depth != a->stream_depth || info->down_rows != down_rows ||
        info->down_mapping != QWEN3MOE_FULL_MOE_Q8_128_DOWN_MAPPING || !info->barrier_timeout_ticks ||
        info->active_blocks_per_multiprocessor <= 0 || info->multiprocessors <= 0 ||
        int64_t(info->active_blocks_per_multiprocessor) * info->multiprocessors < a->grid_blocks) {
        return hipErrorInvalidValue;
    }
    hipError_t error = hipMemsetAsync(a->arrival_counter, 0, sizeof(uint32_t), stream);
    if (error != hipSuccess) return error;
    if (a->down_type == GGML_TYPE_Q6_K) {
        return a->stream_depth == 2 ? submit<true, 2>(*a, info->barrier_timeout_ticks, stream)
                                         : submit<true, 4>(*a, info->barrier_timeout_ticks, stream);
    }
    return a->stream_depth == 2 ? submit<false, 2>(*a, info->barrier_timeout_ticks, stream)
                                     : submit<false, 4>(*a, info->barrier_timeout_ticks, stream);
}

extern "C" hipError_t qwen3moe_full_moe_q8_128_launch(const qwen3moe_full_moe_q8_128_args * a, hipStream_t stream) {
    qwen3moe_full_moe_q8_128_launch_info info{};
    const hipError_t error = qwen3moe_full_moe_q8_128_query(a, &info);
    return error == hipSuccess ? qwen3moe_full_moe_q8_128_enqueue(a, &info, stream) : error;
}
