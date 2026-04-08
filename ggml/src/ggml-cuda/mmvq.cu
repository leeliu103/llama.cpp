#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>

static uint32_t ggml_cuda_q8_0_d_offset_2d(const int64_t ne00, const int64_t nrows) {
    const uint64_t nblocks = (uint64_t) ne00 * (uint64_t) nrows / QK8_0;
    const uint64_t d_offset = nblocks * QK8_0;

    GGML_ASSERT(d_offset <= UINT32_MAX);
    return (uint32_t) d_offset;
}

static uint32_t ggml_cuda_x_soa_aux_offset(const ggml_tensor * src0) {
    if (src0->type == GGML_TYPE_MXFP4) {
        const uint64_t nblocks = (uint64_t) ggml_nelements(src0) / ggml_blck_size(src0->type);
        const uint64_t q_offset = nblocks * (QK_MXFP4 / 2);

        GGML_ASSERT(q_offset <= UINT32_MAX);
        return (uint32_t) q_offset;
    }

    if (ggml_cuda_tensor_uses_q8_0_soa(src0)) {
        return ggml_cuda_q8_0_d_offset_2d(src0->ne[0], ggml_nelements(src0) / src0->ne[0]);
    }

    return 0;
}

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

static __device__ __forceinline__ float vec_dot_mxfp4_soa_q8_1_x4(
        const uint8_t * __restrict__ vx_e,
        const uint8_t * __restrict__ vx_q,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const int block_outer = kby >> 2;
    const int block_inner = kby & 3;
    const block_q8_1_x4 * by = yx4 + block_outer;
    const int * q8 = (const int *) by->qs + block_inner * 8 + iqs;
    const uint8_t * q4 = vx_q + (int64_t) kbx*(QK_MXFP4/2);

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(q4, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = ggml_cuda_e8m0_to_fp32(vx_e[kbx]) * 0.5f * __low2float(by->ds[block_inner]);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_q8_0_soa_q8_1(
        const int8_t * __restrict__ vx_q,
        const ggml_half * __restrict__ vx_d,
        const block_q8_1 * __restrict__ bq8_1,
        const int & kbx, const int & iqs) {
    const int8_t * q8_0 = vx_q + (int64_t) kbx*QK8_0;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(q8_0, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, vx_d[kbx], __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q8_0_soa_q8_1_x4(
        const int8_t * __restrict__ vx_q,
        const ggml_half * __restrict__ vx_d,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const int block_outer = kby >> 2;
    const int block_inner = kby & 3;
    const block_q8_1_x4 * by = yx4 + block_outer;
    const int * q8 = (const int *) by->qs + block_inner * 8 + iqs;
    const int8_t * q8_0 = vx_q + (int64_t) kbx * QK8_0;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(q8_0, iqs + i);
        u[i] = q8[i];
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, vx_d[kbx], __low2half(by->ds[block_inner]));
}

static __device__ __forceinline__ int pack_i8x4(
        const int8_t v0, const int8_t v1, const int8_t v2, const int8_t v3) {
    return
        ((uint32_t) (uint8_t) v0) |
        ((uint32_t) (uint8_t) v1 <<  8) |
        ((uint32_t) (uint8_t) v2 << 16) |
        ((uint32_t) (uint8_t) v3 << 24);
}

struct q4_k_q8_1_x4_rhs {
    int4   q8;
    float2 ds;
};

static __device__ __forceinline__ q4_k_q8_1_x4_rhs load_q4_K_q8_1_x4_rhs(
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int subblock, const int subblock_half) {
    const int block_offset = kby + subblock;
    const int block_outer  = block_offset >> 2;
    const int block_inner  = block_offset & 3;
    const block_q8_1_x4 * by = yx4 + block_outer;
    const int4 * q8 = (const int4 *) (by->qs + block_inner * 8 + subblock_half * 4);

    q4_k_q8_1_x4_rhs rhs;
    rhs.q8 = q8[0];
    rhs.ds = __half22float2(by->ds[block_inner]);
    return rhs;
}

struct q4_k_scale_min {
    uint32_t scale;
    uint32_t min;
};

struct q4_k_block_header {
    half2    dm;
    uint32_t scale0;
    uint32_t scale4;
    uint32_t scale8;
};

static_assert(sizeof(q4_k_block_header) == 16, "Unexpected q4_k block header size");

static __device__ __forceinline__ q4_k_block_header load_q4_K_block_header(
        const block_q4_K * __restrict__ bq4_K) {
    return ((const q4_k_block_header *) bq4_K)[0];
}

static __device__ __forceinline__ q4_k_scale_min load_q4_K_scale_min(
        const q4_k_block_header & header,
        const int subblock) {
    const uint32_t sc_lo = header.scale0;
    const uint32_t mb_lo = header.scale4;
    const uint32_t sc_hi = (header.scale8 & 0x0F0F0F0Fu) | ((header.scale0 & 0xC0C0C0C0u) >> 2);
    const uint32_t mb_hi = ((header.scale8 & 0xF0F0F0F0u) >> 4) | ((header.scale4 & 0xC0C0C0C0u) >> 2);

    const int shift = 8 * (subblock & 3);
    const uint32_t sc_word = subblock < 4 ? sc_lo : sc_hi;
    const uint32_t mb_word = subblock < 4 ? mb_lo : mb_hi;

    q4_k_scale_min scale_min;
    scale_min.scale = (sc_word >> shift) & 0x3Fu;
    scale_min.min   = (mb_word >> shift) & 0x3Fu;
    return scale_min;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_x4_with_rhs(
        const block_q4_K * __restrict__ bq4_K,
        const q4_k_q8_1_x4_rhs & rhs,
        const int subblock, const int iqs_k) {
    const int qs_idx   = (iqs_k / 16) * 8 + (iqs_k % 8);
    const int qs_shift = ((iqs_k % 16) / 8) * 4;
    const uint4 packed_q4 = ((const uint4 *) ((const uint32_t *) bq4_K->qs + qs_idx))[0];
    const q4_k_block_header header = load_q4_K_block_header(bq4_K);

    const int v0 = (packed_q4.x >> qs_shift) & 0x0F0F0F0F;
    const int v1 = (packed_q4.y >> qs_shift) & 0x0F0F0F0F;
    const int v2 = (packed_q4.z >> qs_shift) & 0x0F0F0F0F;
    const int v3 = (packed_q4.w >> qs_shift) & 0x0F0F0F0F;
    const q4_k_scale_min scale_min = load_q4_K_scale_min(header, subblock);

    int q_sum = 0;
    q_sum = ggml_cuda_dp4a(v0, rhs.q8.x, q_sum);
    q_sum = ggml_cuda_dp4a(v1, rhs.q8.y, q_sum);
    q_sum = ggml_cuda_dp4a(v2, rhs.q8.z, q_sum);
    q_sum = ggml_cuda_dp4a(v3, rhs.q8.w, q_sum);

    const float2 dm = __half22float2(header.dm);
    return rhs.ds.x * (dm.x * scale_min.scale) * q_sum - (dm.y * scale_min.min) * (rhs.ds.y * 0.5f);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_x4(
        const void * __restrict__ vbq,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    // Mirror Vulkan MMVQ: each thread handles one 32-value Q4_K subblock and one 16-value half.
    const int subblock = iqs / 4;
    const int subblock_half = (iqs >> 1) & 1;
    const int iqs_k = subblock * 8 + subblock_half * 4;

    const q4_k_q8_1_x4_rhs rhs = load_q4_K_q8_1_x4_rhs(yx4, kby, subblock, subblock_half);
    return vec_dot_q4_K_q8_1_x4_with_rhs(bq4_K, rhs, subblock, iqs_k);
}

static __device__ __forceinline__ float2 vec_dot_q4_K_q8_1_x4_x2(
        const void * __restrict__ vbq_x,
        const void * __restrict__ vbq_gate,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K_x = (const block_q4_K *) vbq_x + kbx;
    const block_q4_K * bq4_K_gate = (const block_q4_K *) vbq_gate + kbx;

    const int subblock = iqs / 4;
    const int subblock_half = (iqs >> 1) & 1;
    const int iqs_k = subblock * 8 + subblock_half * 4;
    const q4_k_q8_1_x4_rhs rhs = load_q4_K_q8_1_x4_rhs(yx4, kby, subblock, subblock_half);

    return make_float2(
        vec_dot_q4_K_q8_1_x4_with_rhs(bq4_K_x, rhs, subblock, iqs_k),
        vec_dot_q4_K_q8_1_x4_with_rhs(bq4_K_gate, rhs, subblock, iqs_k));
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_x4(
        const void * __restrict__ vbq,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int block_offset = kby + bq8_offset + 2 * i;
        const int block_outer  = block_offset >> 2;
        const int block_inner  = block_offset & 3;
        const block_q8_1_x4 * by = yx4 + block_outer;

        const int q8 = by->qs[block_inner * 8 + (iqs % QI8_1)];
        const float d8 = __low2float(by->ds[block_inner]);
        const int sc = scales[4 * i];

        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);

        sumf += d8 * (ggml_cuda_dp4a(vi, q8, 0) * sc);
    }

    return __half2float(bq6_K->d) * sumf;
}

struct q6_k_q8_1_x4_rhs {
    int   q8[4];
    float d8;
};

static __device__ __forceinline__ q6_k_q8_1_x4_rhs load_q6_K_q8_1_x4_rhs(
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int subblock, const int subblock_half) {
    const int block_offset = kby + subblock;
    const int block_outer  = block_offset >> 2;
    const int block_inner  = block_offset & 3;
    const block_q8_1_x4 * by = yx4 + block_outer;
    const int * q8 = (const int *) by->qs + block_inner * 8 + subblock_half * 4;

    q6_k_q8_1_x4_rhs rhs;
    rhs.q8[0] = q8[0];
    rhs.q8[1] = q8[1];
    rhs.q8[2] = q8[2];
    rhs.q8[3] = q8[3];
    rhs.d8 = __low2float(by->ds[block_inner]);
    return rhs;
}

static __device__ __forceinline__ int repack_q6_K_q8_1_x4_chunk(
        const uint16_t ql0, const uint16_t ql1,
        const uint16_t qh0, const uint16_t qh1) {
    const int8_t v0 = (int8_t) (((ql0 >> 0) & 0x0F) | (((qh0 >> 0) & 0x03) << 4)) - 32;
    const int8_t v1 = (int8_t) (((ql0 >> 8) & 0x0F) | (((qh0 >> 8) & 0x03) << 4)) - 32;
    const int8_t v2 = (int8_t) (((ql1 >> 0) & 0x0F) | (((qh1 >> 0) & 0x03) << 4)) - 32;
    const int8_t v3 = (int8_t) (((ql1 >> 8) & 0x0F) | (((qh1 >> 8) & 0x03) << 4)) - 32;
    return pack_i8x4(v0, v1, v2, v3);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_x4_with_rhs(
        const block_q6_K * __restrict__ bq6_K,
        const q6_k_q8_1_x4_rhs & rhs,
        const int subblock, const int subblock_half) {
    const int iqs = subblock_half * 4;
    const int iqs_k = subblock * 8 + iqs;

    const int ql_idx = (iqs_k / 32) * 16 + (iqs_k % 16);
    const int ql_shift = ((iqs_k % 32) / 16) * 4;

    const int qh_idx = (iqs_k / 32) * 8 + iqs;
    const int qh_shift = ((iqs_k % 32) / 8) * 2;

    const uint16_t * ql = (const uint16_t *) bq6_K->ql;
    const uint16_t * qh = (const uint16_t *) bq6_K->qh;

    const int v0 = repack_q6_K_q8_1_x4_chunk(
        (ql[ql_idx * 2 + 0] >> ql_shift) & 0x0F0F,
        (ql[ql_idx * 2 + 1] >> ql_shift) & 0x0F0F,
        (qh[qh_idx * 2 + 0] >> qh_shift) & 0x0303,
        (qh[qh_idx * 2 + 1] >> qh_shift) & 0x0303);
    const int v1 = repack_q6_K_q8_1_x4_chunk(
        (ql[ql_idx * 2 + 2] >> ql_shift) & 0x0F0F,
        (ql[ql_idx * 2 + 3] >> ql_shift) & 0x0F0F,
        (qh[qh_idx * 2 + 2] >> qh_shift) & 0x0303,
        (qh[qh_idx * 2 + 3] >> qh_shift) & 0x0303);
    const int v2 = repack_q6_K_q8_1_x4_chunk(
        (ql[ql_idx * 2 + 4] >> ql_shift) & 0x0F0F,
        (ql[ql_idx * 2 + 5] >> ql_shift) & 0x0F0F,
        (qh[qh_idx * 2 + 4] >> qh_shift) & 0x0303,
        (qh[qh_idx * 2 + 5] >> qh_shift) & 0x0303);
    const int v3 = repack_q6_K_q8_1_x4_chunk(
        (ql[ql_idx * 2 + 6] >> ql_shift) & 0x0F0F,
        (ql[ql_idx * 2 + 7] >> ql_shift) & 0x0F0F,
        (qh[qh_idx * 2 + 6] >> qh_shift) & 0x0303,
        (qh[qh_idx * 2 + 7] >> qh_shift) & 0x0303);

    int q_sum = 0;
    q_sum = ggml_cuda_dp4a(v0, rhs.q8[0], q_sum);
    q_sum = ggml_cuda_dp4a(v1, rhs.q8[1], q_sum);
    q_sum = ggml_cuda_dp4a(v2, rhs.q8[2], q_sum);
    q_sum = ggml_cuda_dp4a(v3, rhs.q8[3], q_sum);

    const float d_scale = __half2float(bq6_K->d) * bq6_K->scales[subblock * 2 + subblock_half];
    return rhs.d8 * d_scale * q_sum;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_x4_packed16(
        const void * __restrict__ vbq,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const int iqs0 = 2 * iqs;
    return vec_dot_q6_K_q8_1_x4(vbq, yx4, kby, kbx, iqs0 + 0) +
           vec_dot_q6_K_q8_1_x4(vbq, yx4, kby, kbx, iqs0 + 1);
}

static __device__ __forceinline__ float2 vec_dot_q6_K_q8_1_x4_packed16_x2(
        const void * __restrict__ vbq_x,
        const void * __restrict__ vbq_gate,
        const block_q8_1_x4 * __restrict__ yx4,
        const int & kby, const int & kbx, const int & iqs) {
    const int iqs0 = 2 * iqs;
    return make_float2(
        vec_dot_q6_K_q8_1_x4(vbq_x, yx4, kby, kbx, iqs0 + 0) +
        vec_dot_q6_K_q8_1_x4(vbq_x, yx4, kby, kbx, iqs0 + 1),
        vec_dot_q6_K_q8_1_x4(vbq_gate, yx4, kby, kbx, iqs0 + 0) +
        vec_dot_q6_K_q8_1_x4(vbq_gate, yx4, kby, kbx, iqs0 + 1));
}

template <bool has_fusion, bool has_gate_fusion, bool has_ids, int nwarps>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q4_K_q8_1_x4(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    static_assert(!has_gate_fusion || has_fusion, "gate fusion requires fusion support");

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    if constexpr (has_ids) {
        channel_x  = ids[channel_dst];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        channel_x  = fastdiv(channel_dst, channel_ratio);
        channel_y  = channel_dst;
        sample_dst = blockIdx.z;
    }

    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    const block_q8_1_x4 * y_row = ((const block_q8_1_x4 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const void * vx_row = ((const block_q4_K *) vx) + kbx_offset;
    const void * vgate_row = nullptr;
    if constexpr (has_gate_fusion) {
        vgate_row = ((const block_q4_K *) fusion.gate) + kbx_offset;
    }

    float tmp = 0.0f;
    float tmp_gate = 0.0f;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

        if constexpr (has_gate_fusion) {
            const float2 dots = vec_dot_q4_K_q8_1_x4_x2(vx_row, vgate_row, y_row, kby, kbx, kqs);
            tmp += dots.x;
            tmp_gate += dots.y;
        } else {
            tmp += vec_dot_q4_K_q8_1_x4(vx_row, y_row, kby, kbx, kqs);
        }
    }

    __shared__ float tmp_shared[(nwarps - 1 > 0) ? nwarps - 1 : 1][warp_size];
    __shared__ float tmp_shared_gate[(has_gate_fusion && (nwarps - 1 > 0)) ? nwarps - 1 : 1][warp_size];
    if constexpr (!has_gate_fusion) {
        (void) tmp_shared_gate;
    }

    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
            if constexpr (has_gate_fusion) {
                tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }

#pragma unroll
        for (int warp = 0; warp < nwarps - 1; ++warp) {
            tmp += tmp_shared[warp][threadIdx.x];
            if constexpr (has_gate_fusion) {
                tmp_gate += tmp_shared_gate[warp][threadIdx.x];
            }
        }
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if constexpr (has_gate_fusion) {
        tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);
    }

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float result = tmp;

        if constexpr (has_fusion) {
            const uint32_t channel_bias = has_ids ? channel_x : channel_dst;

            if (fusion.x_bias != nullptr) {
                const float * x_bias = (const float *) fusion.x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
                result += x_bias[0];
            }

            if constexpr (has_gate_fusion) {
                float gate_value = tmp_gate;
                if (fusion.gate_bias != nullptr) {
                    const float * gate_bias = (const float *) fusion.gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
                    gate_value += gate_bias[0];
                }
                switch (fusion.glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    default:
                        result *= gate_value;
                        break;
                }
            }
        }

        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED(fusion);
    } else if constexpr (!has_gate_fusion) {
        GGML_UNUSED(fusion.gate);
        GGML_UNUSED(fusion.gate_bias);
        GGML_UNUSED(fusion.glu_op);
    }

    if constexpr (!has_ids) {
        GGML_UNUSED(ids);
        GGML_UNUSED(nchannels_y);
    } else {
        GGML_UNUSED(channel_ratio);
    }
    GGML_UNUSED_VARS(stride_col_y, stride_col_dst, ids_stride);
}

template <bool has_fusion, bool has_gate_fusion, bool has_ids, int nwarps>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q6_K_q8_1_x4(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    static_assert(!has_gate_fusion || has_fusion, "gate fusion requires fusion support");

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk;
    constexpr int values_per_thread = 16;
    constexpr int chunks_per_block = qk / values_per_thread;
    constexpr int blocks_per_iter = nwarps * warp_size / chunks_per_block;

    static_assert(chunks_per_block > 0, "invalid q6 chunk count");
    static_assert((nwarps * warp_size) % chunks_per_block == 0, "q6 kernel expects an integral number of blocks per iteration");

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    if constexpr (has_ids) {
        channel_x  = ids[channel_dst];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        channel_x  = fastdiv(channel_dst, channel_ratio);
        channel_y  = channel_dst;
        sample_dst = blockIdx.z;
    }

    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    const block_q8_1_x4 * y_row = ((const block_q8_1_x4 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const void * vx_row = ((const block_q6_K *) vx) + kbx_offset;
    const void * vgate_row = nullptr;
    if constexpr (has_gate_fusion) {
        vgate_row = ((const block_q6_K *) fusion.gate) + kbx_offset;
    }

    float tmp = 0.0f;
    float tmp_gate = 0.0f;

    for (int kbx = tid / chunks_per_block; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = tid % chunks_per_block;

        if constexpr (has_gate_fusion) {
            const float2 dots = vec_dot_q6_K_q8_1_x4_packed16_x2(vx_row, vgate_row, y_row, kby, kbx, kqs);
            tmp += dots.x;
            tmp_gate += dots.y;
        } else {
            tmp += vec_dot_q6_K_q8_1_x4_packed16(vx_row, y_row, kby, kbx, kqs);
        }
    }

    __shared__ float tmp_shared[(nwarps - 1 > 0) ? nwarps - 1 : 1][warp_size];
    __shared__ float tmp_shared_gate[(has_gate_fusion && (nwarps - 1 > 0)) ? nwarps - 1 : 1][warp_size];
    if constexpr (!has_gate_fusion) {
        (void) tmp_shared_gate;
    }

    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
            if constexpr (has_gate_fusion) {
                tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }

#pragma unroll
        for (int warp = 0; warp < nwarps - 1; ++warp) {
            tmp += tmp_shared[warp][threadIdx.x];
            if constexpr (has_gate_fusion) {
                tmp_gate += tmp_shared_gate[warp][threadIdx.x];
            }
        }
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if constexpr (has_gate_fusion) {
        tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);
    }

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float result = tmp;

        if constexpr (has_fusion) {
            const uint32_t channel_bias = has_ids ? channel_x : channel_dst;

            if (fusion.x_bias != nullptr) {
                const float * x_bias = (const float *) fusion.x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
                result += x_bias[0];
            }

            if constexpr (has_gate_fusion) {
                float gate_value = tmp_gate;
                if (fusion.gate_bias != nullptr) {
                    const float * gate_bias = (const float *) fusion.gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
                    gate_value += gate_bias[0];
                }
                switch (fusion.glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    default:
                        result *= gate_value;
                        break;
                }
            }
        }

        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED(fusion);
    } else if constexpr (!has_gate_fusion) {
        GGML_UNUSED(fusion.gate);
        GGML_UNUSED(fusion.gate_bias);
        GGML_UNUSED(fusion.glu_op);
    }

    if constexpr (!has_ids) {
        GGML_UNUSED(ids);
        GGML_UNUSED(nchannels_y);
    } else {
        GGML_UNUSED(channel_ratio);
    }
    GGML_UNUSED_VARS(stride_col_y, stride_col_dst, ids_stride);
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA2) || defined(RDNA3) || defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static int calc_q4_K_q8_1_x4_nwarps(const int cc, const int nrows_x, const int ncols_x) {
    if ((GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) &&
        nrows_x <= 32768 && ncols_x >= 1024) {
        return 2;
    }
    return 1;
}

static int calc_q6_K_q8_1_x4_nwarps(const int cc, const int nrows_x, const int ncols_x) {
    GGML_UNUSED(cc);
    GGML_UNUSED(nrows_x);
    GGML_UNUSED(ncols_x);
    return 1;
}

template <bool has_fusion, bool has_gate_fusion, bool has_ids>
static void launch_mul_mat_vec_q4_K_q8_1_x4(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & generic_block_dims,
        const uint32_t ids_stride, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int nwarps = calc_q4_K_q8_1_x4_nwarps(cc, (int) block_nums.x, (int) ncols_x);

    if (nwarps == 4) {
        const dim3 block_dims(generic_block_dims.x, 4, 1);
        mul_mat_vec_q4_K_q8_1_x4<has_fusion, has_gate_fusion, has_ids, 4><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
        return;
    }

    if (nwarps == 2) {
        const dim3 block_dims(generic_block_dims.x, 2, 1);
        mul_mat_vec_q4_K_q8_1_x4<has_fusion, has_gate_fusion, has_ids, 2><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
        return;
    }

    {
        const dim3 block_dims(generic_block_dims.x, 1, 1);
        mul_mat_vec_q4_K_q8_1_x4<has_fusion, has_gate_fusion, has_ids, 1><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
    }
}

template <bool has_fusion, bool has_gate_fusion, bool has_ids>
static void launch_mul_mat_vec_q6_K_q8_1_x4(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & generic_block_dims,
        const uint32_t ids_stride, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int nwarps = calc_q6_K_q8_1_x4_nwarps(cc, (int) block_nums.x, (int) ncols_x);

    if (nwarps == 2) {
        const dim3 block_dims(generic_block_dims.x, 2, 1);
        mul_mat_vec_q6_K_q8_1_x4<has_fusion, has_gate_fusion, has_ids, 2><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
        return;
    }

    {
        const dim3 block_dims(generic_block_dims.x, 1, 1);
        mul_mat_vec_q6_K_q8_1_x4<has_fusion, has_gate_fusion, has_ids, 1><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
    }
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool has_gate_fusion = false, bool is_multi_token_id = false, bool use_y_q8_1_x4 = false>
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride,
        const uint32_t x_soa_aux_offset) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t token_idx = 0;
    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    static_assert(!has_gate_fusion || has_fusion, "gate fusion requires fusion support");

    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
        channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
        sample_dst = blockIdx.z;
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu = GGML_GLU_OP_SWIGLU;

    if constexpr (has_fusion) {
        use_bias      = fusion.x_bias    != nullptr;
        x_bias        = (const float *) fusion.x_bias;
        if constexpr (has_gate_fusion) {
            vgate         = fusion.gate;
            use_gate_bias = fusion.gate_bias != nullptr;
            gate_bias     = (const float *) fusion.gate_bias;
            active_glu    = fusion.glu_op;
        }
    }

    const uint8_t * vx_e = nullptr;
    const uint8_t * vx_q = nullptr;
    const uint8_t * vg_e = nullptr;
    const uint8_t * vg_q = nullptr;
    const bool use_q8_0_soa = type == GGML_TYPE_Q8_0 && x_soa_aux_offset != 0;
    const bool use_q8_0_soa_x4 = use_q8_0_soa && use_y_q8_1_x4;
    const int8_t * vx_q8_0_q = nullptr;
    const ggml_half * vx_q8_0_d = nullptr;
    const int8_t * vg_q8_0_q = nullptr;
    const ggml_half * vg_q8_0_d = nullptr;
    if constexpr (type == GGML_TYPE_MXFP4) {
        vx_q = (const uint8_t *) vx;
        vx_e = vx_q + x_soa_aux_offset;
        if constexpr (has_fusion && has_gate_fusion) {
            vg_q = (const uint8_t *) vgate;
            vg_e = vg_q + x_soa_aux_offset;
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        if (use_q8_0_soa) {
            vx_q8_0_q = (const int8_t *) vx;
            vx_q8_0_d = (const ggml_half *) (vx_q8_0_q + x_soa_aux_offset);
            if constexpr (has_fusion && has_gate_fusion) {
                vg_q8_0_q = (const int8_t *) vgate;
                vg_q8_0_d = (const ggml_half *) (vg_q8_0_q + x_soa_aux_offset);
            }
        }
    } else {
        GGML_UNUSED(x_soa_aux_offset);
    }


    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[has_gate_fusion ? ncols_dst : 1][has_gate_fusion ? rows_per_cuda_block : 1] = {{0.0f}};

    const block_q8_1 * y = nullptr;
    const block_q8_1_x4 * yx4 = nullptr;
    if constexpr (use_y_q8_1_x4) {
        yx4 = ((const block_q8_1_x4 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
        if constexpr (is_multi_token_id) {
            yx4 += token_idx*stride_col_y;
        }
    } else {
        y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
        if constexpr (is_multi_token_id) {
            y += token_idx*stride_col_y;
        }
    }
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const int kbx_x = kbx_offset + i*stride_row_x + kbx;

                if constexpr (type == GGML_TYPE_MXFP4) {
                    const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                    tmp[j][i] += vec_dot_mxfp4_soa_q8_1_x4(vx_e, vx_q, yb_x4, kby, kbx_x, kqs);
                } else if constexpr (type == GGML_TYPE_Q8_0) {
                    if (use_q8_0_soa_x4) {
                        const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                        tmp[j][i] += vec_dot_q8_0_soa_q8_1_x4(vx_q8_0_q, vx_q8_0_d, yb_x4, kby, kbx_x, kqs);
                    } else {
                        const block_q8_1 * yb = &y[j*stride_col_y + kby];
                        if (use_q8_0_soa) {
                            tmp[j][i] += vec_dot_q8_0_soa_q8_1(vx_q8_0_q, vx_q8_0_d, yb, kbx_x, kqs);
                        } else {
                            tmp[j][i] += vec_dot_q_cuda(vx, yb, kbx_x, kqs);
                        }
                    }
                } else if constexpr (type == GGML_TYPE_Q4_K) {
                    if constexpr (use_y_q8_1_x4) {
                        const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                        if constexpr (has_fusion && has_gate_fusion) {
                            const float2 dots = vec_dot_q4_K_q8_1_x4_x2(vx, vgate, yb_x4, kby, kbx_x, kqs);
                            tmp[j][i] += dots.x;
                            tmp_gate[j][i] += dots.y;
                            continue;
                        } else {
                            tmp[j][i] += vec_dot_q4_K_q8_1_x4(vx, yb_x4, kby, kbx_x, kqs);
                        }
                    } else {
                        const block_q8_1 * yb = &y[j*stride_col_y + kby];
                        tmp[j][i] += vec_dot_q_cuda(vx, yb, kbx_x, kqs);
                    }
                } else if constexpr (type == GGML_TYPE_Q6_K) {
                    if constexpr (use_y_q8_1_x4) {
                        const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                        tmp[j][i] += vec_dot_q6_K_q8_1_x4(vx, yb_x4, kby, kbx_x, kqs);
                    } else {
                        const block_q8_1 * yb = &y[j*stride_col_y + kby];
                        tmp[j][i] += vec_dot_q_cuda(vx, yb, kbx_x, kqs);
                    }
                } else {
                    const block_q8_1 * yb = &y[j*stride_col_y + kby];
                    tmp[j][i] += vec_dot_q_cuda(vx, yb, kbx_x, kqs);
                }
                if constexpr (has_fusion && has_gate_fusion) {
                    if constexpr (type == GGML_TYPE_MXFP4) {
                        const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                        tmp_gate[j][i] += vec_dot_mxfp4_soa_q8_1_x4(vg_e, vg_q, yb_x4, kby, kbx_x, kqs);
                    } else if constexpr (type == GGML_TYPE_Q8_0) {
                        if (use_q8_0_soa_x4) {
                            const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                            tmp_gate[j][i] += vec_dot_q8_0_soa_q8_1_x4(vg_q8_0_q, vg_q8_0_d, yb_x4, kby, kbx_x, kqs);
                        } else {
                            const block_q8_1 * yb = &y[j*stride_col_y + kby];
                            if (use_q8_0_soa) {
                                tmp_gate[j][i] += vec_dot_q8_0_soa_q8_1(vg_q8_0_q, vg_q8_0_d, yb, kbx_x, kqs);
                            } else {
                                tmp_gate[j][i] += vec_dot_q_cuda(vgate, yb, kbx_x, kqs);
                            }
                        }
                    } else if constexpr (type == GGML_TYPE_Q4_K) {
                        if constexpr (use_y_q8_1_x4) {
                            const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                            tmp_gate[j][i] += vec_dot_q4_K_q8_1_x4(vgate, yb_x4, kby, kbx_x, kqs);
                        } else {
                            const block_q8_1 * yb = &y[j*stride_col_y + kby];
                            tmp_gate[j][i] += vec_dot_q_cuda(vgate, yb, kbx_x, kqs);
                        }
                    } else if constexpr (type == GGML_TYPE_Q6_K) {
                        if constexpr (use_y_q8_1_x4) {
                            const block_q8_1_x4 * yb_x4 = &yx4[j*stride_col_y];
                            tmp_gate[j][i] += vec_dot_q6_K_q8_1_x4(vgate, yb_x4, kby, kbx_x, kqs);
                        } else {
                            const block_q8_1 * yb = &y[j*stride_col_y + kby];
                            tmp_gate[j][i] += vec_dot_q_cuda(vgate, yb, kbx_x, kqs);
                        }
                    } else {
                        const block_q8_1 * yb = &y[j*stride_col_y + kby];
                        tmp_gate[j][i] += vec_dot_q_cuda(vgate, yb, kbx_x, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_gate_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][has_gate_fusion ? ncols_dst : 1][has_gate_fusion ? rows_per_cuda_block : 1][warp_size];
    if constexpr (!has_gate_fusion) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_gate_fusion) {
                    tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    if constexpr (is_multi_token_id) {
        dst += token_idx*stride_col_dst;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_gate_fusion) {
                    tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_gate_fusion) {
                tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if constexpr (has_gate_fusion) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    } else if constexpr (!has_gate_fusion) {
        GGML_UNUSED_VARS(use_gate_bias, active_glu, gate_bias, tmp_gate);
    }
}

static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id) {
    const int64_t nblocks = (nrows_x + calc_rows_per_block(ncols_dst, table_id) - 1) / calc_rows_per_block(ncols_dst, table_id);
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, calc_nwarps(ncols_dst, table_id), 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool use_y_q8_1_x4, bool is_multi_token_id = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride,
        const uint32_t x_soa_aux_offset, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (type == GGML_TYPE_Q4_K && c_ncols_dst == 1 && use_y_q8_1_x4 && !is_multi_token_id) {
        if (ids != nullptr) {
            if constexpr (c_ncols_dst == 1) {
                if (has_fusion) {
                    if (fusion.gate != nullptr) {
                        launch_mul_mat_vec_q4_K_q8_1_x4<true, true, true>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, ids_stride, stream);
                    } else {
                        launch_mul_mat_vec_q4_K_q8_1_x4<true, false, true>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, ids_stride, stream);
                    }
                } else {
                    launch_mul_mat_vec_q4_K_q8_1_x4<false, false, true>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                        block_nums, block_dims, ids_stride, stream);
                }
                return;
            }
        } else {
            if (has_fusion) {
                if (fusion.gate != nullptr) {
                    launch_mul_mat_vec_q4_K_q8_1_x4<true, true, false>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                        block_nums, block_dims, ids_stride, stream);
                } else {
                    launch_mul_mat_vec_q4_K_q8_1_x4<true, false, false>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                        block_nums, block_dims, ids_stride, stream);
                }
            } else {
                launch_mul_mat_vec_q4_K_q8_1_x4<false, false, false>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                    block_nums, block_dims, ids_stride, stream);
            }
            return;
        }
    }
    if constexpr (false && type == GGML_TYPE_Q6_K && c_ncols_dst == 1 && use_y_q8_1_x4 && !is_multi_token_id) {
        if (!has_fusion && ids == nullptr) {
            launch_mul_mat_vec_q6_K_q8_1_x4<false, false, false>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                block_nums, block_dims, ids_stride, stream);
            return;
        }
    }

    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            if (fusion.gate != nullptr) {
                mul_mat_vec_q<type, c_ncols_dst, true, true, is_multi_token_id, use_y_q8_1_x4><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset);
            } else {
                mul_mat_vec_q<type, c_ncols_dst, true, false, is_multi_token_id, use_y_q8_1_x4><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset);
            }
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, c_ncols_dst, false, false, is_multi_token_id, use_y_q8_1_x4><<<block_nums, block_dims, nbytes_shared, stream>>>
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset);
}

template <ggml_type type, bool use_y_q8_1_x4>
static void mul_mat_vec_q_switch_ncols_dst_impl(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride,
        const uint32_t x_soa_aux_offset, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, ncols_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4, true>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
             dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, use_y_q8_1_x4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, x_soa_aux_offset, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride,
        const uint32_t x_soa_aux_offset, const bool y_q8_1_x4, cudaStream_t stream) {
    if constexpr (type == GGML_TYPE_MXFP4) {
        GGML_ASSERT(y_q8_1_x4);
        mul_mat_vec_q_switch_ncols_dst_impl<type, true>(
            vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
            ids_stride, x_soa_aux_offset, stream);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0) {
        if (y_q8_1_x4) {
            mul_mat_vec_q_switch_ncols_dst_impl<type, true>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, x_soa_aux_offset, stream);
        } else {
            mul_mat_vec_q_switch_ncols_dst_impl<type, false>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, x_soa_aux_offset, stream);
        }
    } else {
        GGML_ASSERT(!y_q8_1_x4);
        mul_mat_vec_q_switch_ncols_dst_impl<type, false>(
            vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
            ids_stride, x_soa_aux_offset, stream);
    }
}

static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride,
        const uint32_t x_soa_aux_offset, const bool y_q8_1_x4, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, x_soa_aux_offset, y_q8_1_x4, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            if (ggml_cuda_tensor_uses_q8_0_soa(src0)) {
                GGML_ASSERT(ggml_cuda_tensor_uses_q8_0_soa(fusion->gate));
            }
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const bool use_y_q8_1_x4 = src0->type == GGML_TYPE_Q4_K ||
        src0->type == GGML_TYPE_Q6_K ||
        src0->type == GGML_TYPE_MXFP4 ||
        (src0->type == GGML_TYPE_Q8_0 && ggml_cuda_tensor_uses_q8_0_soa(src0));
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        if (use_y_q8_1_x4) {
            quantize_row_q8_1_x4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        } else {
            quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = use_y_q8_1_x4
        ? (ne10_padded / (4 * QK8_1))
        : (ne10_padded / QK8_1);
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;
    const uint32_t x_soa_aux_offset = ggml_cuda_x_soa_aux_offset(src0);

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, x_soa_aux_offset, use_y_q8_1_x4, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    GGML_ASSERT(src0->type != GGML_TYPE_MXFP4 && "MXFP4 SoA split helper not wired yet");

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0 /* x_soa_aux_offset */, false /* y_q8_1_x4 */, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
