// Derived from the measured private one-head prefix, source SHA256
// b5853502ce81569a08231b074d7c5c533377762c630c3cb39453fd0a0641e6ef.
// Valid-path arithmetic/diagnostic stores are retained. Production changes:
// token-sticky status (no prefix reset), neutral Q on prior/local error, and
// removal of the losing tiled launch. Native stream slice is embedded below.
#include "qwen3moe-qkv-prefix-q8-128.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <array>

namespace {
constexpr int hidden=4096, selected=8, wave_size=32, threads=256, head_width=128;
// Exact native stream/quantizer slice from the frozen split-MoE implementation.
// See provenance.json. The consuming translation unit supplies hidden=4096,
// selected=8 and wave_size=32. No expert kernel or global barrier is included.
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


struct shared_data {
    Smem dot;
    float head[128];
    float rms_sums[8];
    int64_t k_slot, v_slot;
};

// Stock NeoX/YaRN operation order, adapted from the independently checked
// q-head-tail source identified in provenance.json.
__device__ void rotate_pair(float x0,float x1,int pair,int32_t position,
                           const qwen3moe_qkv_prefix_rope & a,float & y0,float & y1) {
    const float theta_base=position*powf(a.theta_scale,float(pair));
    const float theta_interp=a.freq_scale*theta_base;
    float theta=theta_interp,mscale=a.attn_factor;
    if(a.ext_factor!=0.0f) {
        const float y=(pair-a.corr_dims[0])/max(0.001f,a.corr_dims[1]-a.corr_dims[0]);
        const float ramp_mix=(1.0f-min(1.0f,max(0.0f,y)))*a.ext_factor;
        theta=theta_interp*(1-ramp_mix)+theta_base*ramp_mix;
        mscale*=1.0f+0.1f*logf(1.0f/a.freq_scale);
    }
    const float c=cosf(theta)*mscale,s=sinf(theta)*mscale;
    y0=x0*c-x1*s; y1=x0*s+x1*c;
}

__device__ bool live_slots(const qwen3moe_qkv_prefix_q8_128_args & a,shared_data & s) {
    if(threadIdx.x==0) {s.k_slot=a.k_slot[0];s.v_slot=a.v_slot[0];}
    __syncthreads();
    const bool ok=s.k_slot>=0&&s.k_slot<a.cache_rows&&s.v_slot>=0&&s.v_slot<a.cache_rows;
    if(!ok&&threadIdx.x==0)set_error(a.status,QWEN3MOE_QKV_PREFIX_BAD_SLOT);
    return ok; // All threads take the same return, before any address/store.
}

__device__ bool prepare_input(const qwen3moe_qkv_prefix_q8_128_args & a,shared_data & s) {
    const unsigned lane=threadIdx.x%32,wave=threadIdx.x/32;
    float values[2][8];load_group(a.input,wave,values[0]);load_group(a.input,wave+8,values[1]);
    float sum=0.0f;
    for(int group=0;group<2;++group)for(int i=0;i<8;++i)sum=fmaf(values[group][i],values[group][i],sum);
    sum=warp_reduce_sum<32>(sum);
    if(lane==0)s.dot.part[wave][0][0]=sum;
    __syncthreads();
    if(threadIdx.x==0) {
        float total=0;for(int w=0;w<8;++w)total+=s.dot.part[w][0][0];
        s.dot.part[0][1][0]=rsqrtf(total/4096.0f+a.input_epsilon);
    }
    __syncthreads();
    const float inverse=s.dot.part[0][1][0];
    bool valid=isfinite(inverse);
    for(int group=0;group<2;++group)for(int i=0;i<8;++i) {
        const int col=(wave+group*8)*256+lane*8+i;
        values[group][i]=(values[group][i]*inverse)*a.input_gamma[col];
        valid=valid&&isfinite(values[group][i]);
    }
    if(!__syncthreads_and(valid)) {if(threadIdx.x==0)set_error(a.status,QWEN3MOE_QKV_PREFIX_NONFINITE);return false;}
    quantize_group(s.dot,wave,values[0]);quantize_group(s.dot,wave+8,values[1]);
    __syncthreads();
    if(blockIdx.x==0) {
        for(int group=0;group<2;++group)for(int i=0;i<8;++i) {
            const int col=(wave+group*8)*256+lane*8+i;a.normalized[col]=values[group][i];
        }
        for(int i=threadIdx.x;i<4096;i+=256)a.xq[i]=s.dot.xq[i];
        if(threadIdx.x<32)a.xd[threadIdx.x]=s.dot.xd[threadIdx.x];
    }
    return true;
}

struct head_task {int kind,head;};
__device__ head_task task(const qwen3moe_qkv_prefix_q8_128_args & a,int index) {
    if(index<a.q_heads)return {0,index};index-=a.q_heads;
    if(index<a.kv_heads)return {1,index};return {2,index-a.kv_heads};
}
__device__ float * projected(const qwen3moe_qkv_prefix_q8_128_args & a,int kind) {
    return kind==0?a.q_projected:kind==1?a.k_projected:a.v_projected;
}
template<bool q6>
__device__ void project_tile(const qwen3moe_qkv_prefix_q8_128_args & a,shared_data & s,head_task h,int tile) {
    const unsigned lane=threadIdx.x%32,wave=threadIdx.x/32;
    const unsigned row=h.head*128+tile*32+lane;
    const void * weight=h.kind==0?a.wq:h.kind==1?a.wk:a.wv;
    const uint64_t stride=h.kind==0?a.q_stride:h.kind==1?a.k_stride:a.v_stride;
    Stream<2,4,q6?3u:0u,false,1> stream;
    native_set(stream,0,weight,stride,0,0,row,wave,wave);
    native_set(stream,1,weight,stride,0,0,row,wave+8,wave+8);
    stream.pre();float dots[2];stream.run(s.dot,dots);
    __syncthreads();s.dot.part[wave][0][lane]=dots[0]+dots[1];__syncthreads();
    if(wave==0) {
        float value=0;for(int w=0;w<8;++w)value+=s.dot.part[w][0][lane];
        s.head[tile*32+lane]=value;projected(a,h.kind)[row]=value;
    }
    __syncthreads();
}

__device__ void finish_head(const qwen3moe_qkv_prefix_q8_128_args & a,shared_data & s,head_task h) {
    const int tid=threadIdx.x;
    const bool valid=tid>=128||isfinite(s.head[tid]);
    if(!__syncthreads_and(valid)) {if(tid==0)set_error(a.status,QWEN3MOE_QKV_PREFIX_NONFINITE);return;}
    if(h.kind==2) {
        const half value=tid<128?__float2half(s.head[tid]):__float2half(0.0f);
        if(!__syncthreads_and(tid>=128||isfinite(__half2float(value)))) {
            if(tid==0)set_error(a.status,QWEN3MOE_QKV_PREFIX_NONFINITE);return;
        }
        if(tid<128) {
            auto * row=reinterpret_cast<ggml_fp16_t *>(reinterpret_cast<uint8_t *>(a.v_cache)+uint64_t(s.v_slot)*a.v_row_bytes);
            reinterpret_cast<half *>(row)[h.head*128+tid]=value;
        }
        return;
    }
    float sum=tid<128?s.head[tid]*s.head[tid]:0.0f;
    sum=block_reduce<block_reduce_method::SUM,256>(sum,s.rms_sums);
    const qwen3moe_qkv_prefix_rope & rope=h.kind==0?a.q_rope:a.k_rope;
    const float scale=rsqrtf(sum/128+rope.epsilon);
    const float * gamma=h.kind==0?a.q_gamma:a.k_gamma;
    const int32_t position=a.position[0];
    float y0=0,y1=0;half h0=__float2half(0.0f),h1=h0;bool output_valid=true;
    if(tid<64) {
        const float x0=scale*s.head[tid]*gamma[tid],x1=scale*s.head[tid+64]*gamma[tid+64];
        rotate_pair(x0,x1,tid,position,rope,y0,y1);
        output_valid=isfinite(y0)&&isfinite(y1);
        if(h.kind==1) {h0=__float2half(y0);h1=__float2half(y1);output_valid=output_valid&&isfinite(__half2float(h0))&&isfinite(__half2float(h1));}
    }
    if(!__syncthreads_and(output_valid)) {if(tid==0)set_error(a.status,QWEN3MOE_QKV_PREFIX_NONFINITE);return;}
    if(tid<64) {
        if(h.kind==0) {a.q_output[h.head*128+tid]=y0;a.q_output[h.head*128+tid+64]=y1;}
        else {
            a.k_rotated[h.head*128+tid]=y0;a.k_rotated[h.head*128+tid+64]=y1;
            auto * row=reinterpret_cast<half *>(reinterpret_cast<uint8_t *>(a.k_cache)+uint64_t(s.k_slot)*a.k_row_bytes);
            row[h.head*128+tid]=h0;row[h.head*128+tid+64]=h1;
        }
    }
}

template<bool vq6>
__launch_bounds__(256)
__global__ void qwen3moe_qkv_prefix_one_head(qwen3moe_qkv_prefix_q8_128_args a) {
    __shared__ shared_data s;
    const head_task h=task(a,blockIdx.x);
    // A previous/local error must not leave stale or poisoned Q for the stock
    // FA/O and existing collectives that still need to drain. Cache contents
    // remain untouched on prior error or invalid slots. Metadata/ranges were
    // admitted before launch, so this bounded per-head Q store is always safe.
    if(h.kind==0&&threadIdx.x<128)a.q_output[h.head*128+threadIdx.x]=0.0f;
    if(threadIdx.x==0)s.dot.proceed=load_agent(a.status)==0;
    __syncthreads();
    if(!s.dot.proceed||!live_slots(a,s)||!prepare_input(a,s))return;
    for(int tile=0;tile<4;++tile) {
        if(vq6&&h.kind==2)project_tile<true>(a,s,h,tile);else project_tile<false>(a,s,h,tile);
    }
    finish_head(a,s,h);
}
struct region {const void * pointer;size_t size,alignment;};
bool valid_region(region r) {const auto p=reinterpret_cast<uintptr_t>(r.pointer);return p&&r.size&&p%r.alignment==0&&p<=UINTPTR_MAX-r.size;}
bool disjoint(region a,region b) {
    const auto x=reinterpret_cast<uintptr_t>(a.pointer),y=reinterpret_cast<uintptr_t>(b.pointer);
    return x+a.size<=y||y+b.size<=x;
}
bool rope_ok(const qwen3moe_qkv_prefix_rope & r) {
    return std::isfinite(r.epsilon)&&r.epsilon>0&&std::isfinite(r.theta_scale)&&r.theta_scale>0&&
        std::isfinite(r.freq_scale)&&r.freq_scale>0&&std::isfinite(r.ext_factor)&&r.ext_factor>=0&&
        std::isfinite(r.attn_factor)&&std::isfinite(r.corr_dims[0])&&std::isfinite(r.corr_dims[1]);
}
}

bool qwen3moe_qkv_prefix_q8_128_supported(const qwen3moe_qkv_prefix_q8_128_args * a) {
    if(!a||(a->q_heads!=16&&a->q_heads!=32)||a->kv_heads!=a->q_heads/16||
       (a->v_q6!=0&&a->v_q6!=1)||!std::isfinite(a->input_epsilon)||a->input_epsilon<=0||
       !rope_ok(a->q_rope)||!rope_ok(a->k_rope)||a->cache_rows<=0||a->cache_rows>32768)return false;
    const uint64_t qrow=16*sizeof(block_q4_K),vrow=16*(a->v_q6?sizeof(block_q6_K):sizeof(block_q4_K));
    // This initial fixture supports exactly the native contiguous row pitches.
    if(a->q_stride!=qrow||a->k_stride!=qrow||a->v_stride!=vrow||
       a->q_weight_bytes!=qrow*a->q_heads*128||a->k_weight_bytes!=qrow*a->kv_heads*128||
       a->v_weight_bytes!=vrow*a->kv_heads*128||a->k_row_bytes!=uint64_t(a->kv_heads*128*2)||
       a->v_row_bytes!=a->k_row_bytes||a->k_cache_bytes!=a->k_row_bytes*a->cache_rows||
       a->v_cache_bytes!=a->v_row_bytes*a->cache_rows)return false;
    const std::array<region,10> reads={{{a->wq,a->q_weight_bytes,2},{a->wk,a->k_weight_bytes,2},{a->wv,a->v_weight_bytes,2},
        {a->input,4096*4,4},{a->input_gamma,4096*4,4},{a->q_gamma,128*4,4},{a->k_gamma,128*4,4},
        {a->position,4,4},{a->k_slot,8,8},{a->v_slot,8,8}}};
    const size_t qb=size_t(a->q_heads)*128*4,kb=size_t(a->kv_heads)*128*4;
    const std::array<region,11> writes={{{a->normalized,4096*4,4},{a->xq,4096,16},{a->xd,32*4,4},
        {a->q_projected,qb,4},{a->k_projected,kb,4},{a->v_projected,kb,4},{a->q_output,qb,4},{a->k_rotated,kb,4},
        {a->k_cache,a->k_cache_bytes,2},{a->v_cache,a->v_cache_bytes,2},{a->status,4,4}}};
    for(auto r:reads)if(!valid_region(r))return false;
    for(auto w:writes)if(!valid_region(w))return false;
    if(!disjoint(reads[8],reads[9])||!disjoint(reads[7],reads[8])||!disjoint(reads[7],reads[9]))return false;
    for(size_t i=0;i<writes.size();++i) {
        for(auto r:reads)if(!disjoint(writes[i],r))return false;
        for(size_t j=0;j<i;++j)if(!disjoint(writes[i],writes[j]))return false;
    }
    return true;
}
hipError_t qwen3moe_qkv_prefix_q8_128_enqueue(const qwen3moe_qkv_prefix_q8_128_args * a,hipStream_t stream) {
    if(!qwen3moe_qkv_prefix_q8_128_supported(a)||!stream)return hipErrorInvalidValue;
    const int heads=a->q_heads+2*a->kv_heads;
    if(a->v_q6)qwen3moe_qkv_prefix_one_head<true><<<heads,256,0,stream>>>(*a);
    else qwen3moe_qkv_prefix_one_head<false><<<heads,256,0,stream>>>(*a);
    return hipGetLastError();
}
