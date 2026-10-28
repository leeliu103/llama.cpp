#include "qwen3moe-gate-up.h"

#include "unary.cuh"
#include "vecdotq.cuh"

#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int wave_size = 32;
constexpr int waves_per_block = 8;
constexpr int threads_per_block = wave_size * waves_per_block;
constexpr int max_input_size = 8192;

template<int rows_per_wave>
__launch_bounds__(threads_per_block)
__global__ void gate_up_q4_k(qwen3moe_gate_up_q4_k_args a) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto * q8 = reinterpret_cast<block_q8_1 *>(storage);
    const int lane = threadIdx.x % wave_size;
    const int wave = threadIdx.x / wave_size;
    const int slot = blockIdx.y;
    const int expert = a.selected_experts[slot];
    const int row0 = (blockIdx.x * waves_per_block + wave) * rows_per_wave;
    auto * output = reinterpret_cast<float *>(reinterpret_cast<char *>(a.output) + slot * a.output_expert_stride);

    if (expert < 0 || expert >= a.expert_count) {
        if (lane == 0) {
            for (int r = 0; r < rows_per_wave && row0 + r < a.output_rows; ++r) {
                output[row0 + r] = NAN;
            }
        }
        return;
    }

    // Match quantize_q8_1, including FP32 division/rounding and FP16 scale/sum.
    for (int i = threadIdx.x; i < a.input_size; i += threads_per_block) {
        const float x = a.input[i];
        const float amax = warp_reduce_max<QK8_1>(fabsf(x));
        const float sum = warp_reduce_sum<QK8_1>(x);
        const float d = amax / 127.0f;
        q8[i / QK8_1].qs[lane] = amax == 0.0f ? 0 : roundf(x / d);
        if (lane == 0) {
            q8[i / QK8_1].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int lanes_per_quant_block = QI4_K / vdr;
    constexpr int quant_blocks_per_iter = wave_size / lanes_per_quant_block;
    const int quant_blocks = a.input_size / QK_K;
    const int kqs = vdr * (lane % lanes_per_quant_block);

    const char * gate = static_cast<const char *>(a.gate) + expert * a.gate_expert_stride;
    const char * up = static_cast<const char *>(a.up) + expert * a.up_expert_stride;
    float g[rows_per_wave] = {};
    float u[rows_per_wave] = {};
    for (int k = lane / lanes_per_quant_block; k < quant_blocks; k += quant_blocks_per_iter) {
        const block_q8_1 * x = q8 + k * (QK_K / QK8_1);
#pragma unroll
        for (int r = 0; r < rows_per_wave; ++r) {
            if (row0 + r < a.output_rows) {
                g[r] += vec_dot_q4_K_q8_1(gate + (row0 + r) * a.gate_row_stride, x, k, kqs);
                u[r] += vec_dot_q4_K_q8_1(up + (row0 + r) * a.up_row_stride, x, k, kqs);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_wave; ++r) {
        g[r] = warp_reduce_sum<wave_size>(g[r]);
        u[r] = warp_reduce_sum<wave_size>(u[r]);
        if (lane == 0 && row0 + r < a.output_rows) {
            output[row0 + r] = u[r] * ggml_cuda_op_silu_single(g[r]);
        }
    }
}

bool valid_matrix_strides(uint64_t row_stride, uint64_t expert_stride, uint64_t row_bytes,
                          int rows, int experts) {
    if (row_stride < row_bytes || row_stride % alignof(uint32_t) != 0 ||
        expert_stride % alignof(uint32_t) != 0 || row_stride > uint64_t(PTRDIFF_MAX) / rows) {
        return false;
    }
    const uint64_t matrix_bytes = row_stride * (rows - 1) + row_bytes;
    return expert_stride >= matrix_bytes && expert_stride <= uint64_t(PTRDIFF_MAX) / experts;
}

template<int rows_per_wave>
hipError_t launch(const qwen3moe_gate_up_q4_k_args & a, hipStream_t stream) {
    constexpr int rows_per_block = waves_per_block * rows_per_wave;
    const dim3 grid((a.output_rows + rows_per_block - 1) / rows_per_block, a.selected_count);
    const size_t shared_bytes = size_t(a.input_size / QK8_1) * sizeof(block_q8_1);
    hipLaunchKernelGGL(gate_up_q4_k<rows_per_wave>, grid, threads_per_block, shared_bytes, stream, a);
    return hipGetLastError();
}

} // namespace

extern "C" int qwen3moe_gate_up_q4_k_supported(const qwen3moe_gate_up_q4_k_args * a) {
    if (!a || !a->gate || !a->up || !a->input || !a->selected_experts || !a->output ||
        a->input_size <= 0 || a->input_size > max_input_size || a->input_size % QK_K != 0 ||
        a->output_rows <= 0 || a->output_rows > 8192 || a->expert_count <= 0 || a->expert_count > 128 ||
        a->selected_count <= 0 || a->selected_count > a->expert_count ||
        (a->rows_per_wave != 1 && a->rows_per_wave != 2 && a->rows_per_wave != 4 &&
         a->rows_per_wave != 8 && a->rows_per_wave != 16)) {
        return 0;
    }
    for (const void * p : {a->gate, a->up, static_cast<const void *>(a->input),
                          static_cast<const void *>(a->selected_experts), static_cast<const void *>(a->output)}) {
        if (reinterpret_cast<uintptr_t>(p) % alignof(uint32_t) != 0) {
            return 0;
        }
    }
    const uint64_t row_bytes = uint64_t(a->input_size / QK_K) * sizeof(block_q4_K);
    return valid_matrix_strides(a->gate_row_stride, a->gate_expert_stride, row_bytes, a->output_rows, a->expert_count) &&
           valid_matrix_strides(a->up_row_stride, a->up_expert_stride, row_bytes, a->output_rows, a->expert_count) &&
           a->output_expert_stride >= uint64_t(a->output_rows) * sizeof(float) &&
           a->output_expert_stride % sizeof(float) == 0 &&
           a->output_expert_stride <= uint64_t(PTRDIFF_MAX) / a->selected_count;
}

extern "C" hipError_t qwen3moe_gate_up_q4_k_launch(const qwen3moe_gate_up_q4_k_args * a, hipStream_t stream) {
    if (!qwen3moe_gate_up_q4_k_supported(a)) {
        return hipErrorInvalidValue;
    }
    switch (a->rows_per_wave) {
        case 1: return launch<1>(*a, stream);
        case 2: return launch<2>(*a, stream);
        case 4: return launch<4>(*a, stream);
        case 8: return launch<8>(*a, stream);
        case 16: return launch<16>(*a, stream);
        default: return hipErrorInvalidValue;
    }
}
