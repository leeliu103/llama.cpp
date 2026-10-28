#include "qwen3moe-down.h"

#include "vecdotq.cuh"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int wave_size = 32;
constexpr int max_selected = 8;
constexpr int threads_per_block = wave_size * max_selected;

template<bool q6, int rows_per_block>
__launch_bounds__(threads_per_block)
__global__ void down_weighted_sum(qwen3moe_down_args a) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto * q8 = reinterpret_cast<block_q8_1 *>(storage);
    __shared__ float weighted[max_selected][rows_per_block];
    const int lane = threadIdx.x % wave_size;
    const int slot = threadIdx.x / wave_size;
    const int row0 = blockIdx.x * rows_per_block;
    const int q8_blocks = a.input_size / QK8_1;
    const int expert = slot < a.selected_count ? a.selected_experts[slot] : -1;
    const bool valid = slot < a.selected_count && expert >= 0 && expert < a.expert_count;
    block_q8_1 * xq = q8 + slot * q8_blocks;

    if (valid) {
        const float * input = reinterpret_cast<const float *>(reinterpret_cast<const char *>(a.input) + slot * a.input_expert_stride);
        for (int i = lane; i < a.input_size; i += wave_size) {
            const float x = input[i];
            const float amax = warp_reduce_max<QK8_1>(fabsf(x));
            const float sum = warp_reduce_sum<QK8_1>(x);
            const float d = amax / 127.0f;
            xq[i / QK8_1].qs[lane] = amax == 0.0f ? 0 : roundf(x / d);
            if (lane == 0) {
                xq[i / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
    __syncthreads();

    constexpr int vdr = q6 ? VDR_Q6_K_Q8_1_MMVQ : VDR_Q4_K_Q8_1_MMVQ;
    constexpr int qi = q6 ? QI6_K : QI4_K;
    constexpr int lanes_per_quant_block = qi / vdr;
    constexpr int quant_blocks_per_iter = wave_size / lanes_per_quant_block;
    const int kqs = vdr * (lane % lanes_per_quant_block);
    float values[rows_per_block] = {};
    if (valid) {
        const char * matrix = static_cast<const char *>(a.down) + expert * a.down_expert_stride;
        for (int k = lane / lanes_per_quant_block; k < a.input_size / QK_K; k += quant_blocks_per_iter) {
            const block_q8_1 * x = xq + k * (QK_K / QK8_1);
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (row0 + r < a.output_rows) {
                    const void * row = matrix + (row0 + r) * a.down_row_stride;
                    if constexpr (q6) {
                        values[r] += vec_dot_q6_K_q8_1(row, x, k, kqs);
                    } else {
                        values[r] += vec_dot_q4_K_q8_1(row, x, k, kqs);
                    }
                }
            }
        }
    }
    const float weight = valid ? *reinterpret_cast<const float *>(reinterpret_cast<const char *>(a.routing_weights) + slot * a.routing_weight_stride) : 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        values[r] = warp_reduce_sum<wave_size>(values[r]);
        if (lane == 0) {
            // Materialize the product before the ordered adds, as in the stock graph.
            weighted[slot][r] = valid ? values[r] * weight : (slot < a.selected_count ? NAN : 0.0f);
        }
    }
    __syncthreads();
    if (threadIdx.x < rows_per_block && row0 + threadIdx.x < a.output_rows) {
        const int r = threadIdx.x;
        float sum = weighted[0][r];
        for (int e = 1; e < a.selected_count; ++e) {
            sum += weighted[e][r];
        }
        a.output[row0 + r] = sum;
    }
}

template<bool q6, int rows_per_block>
hipError_t launch(const qwen3moe_down_args & a, hipStream_t stream) {
    const dim3 grid((a.output_rows + rows_per_block - 1) / rows_per_block);
    const size_t shared_bytes = size_t(max_selected) * (a.input_size / QK8_1) * sizeof(block_q8_1);
    hipLaunchKernelGGL((down_weighted_sum<q6, rows_per_block>), grid, threads_per_block, shared_bytes, stream, a);
    return hipGetLastError();
}

template<bool q6>
hipError_t dispatch(const qwen3moe_down_args & a, hipStream_t stream) {
    switch (a.rows_per_block) {
        case 8: return launch<q6, 8>(a, stream);
        case 16: return launch<q6, 16>(a, stream);
        case 32: return launch<q6, 32>(a, stream);
        default: return hipErrorInvalidValue;
    }
}

} // namespace

extern "C" int qwen3moe_down_supported(const qwen3moe_down_args * a) {
    if (!a || !a->down || !a->input || !a->selected_experts || !a->routing_weights || !a->output ||
        (a->weight_type != GGML_TYPE_Q4_K && a->weight_type != GGML_TYPE_Q6_K) ||
        a->input_size <= 0 || a->input_size > 2048 || a->input_size % QK_K != 0 ||
        a->output_rows <= 0 || a->output_rows > 8192 || a->expert_count <= 0 || a->expert_count > 128 ||
        a->selected_count <= 0 || a->selected_count > max_selected || a->selected_count > a->expert_count ||
        (a->rows_per_block != 8 && a->rows_per_block != 16 && a->rows_per_block != 32)) {
        return 0;
    }
    for (const void * p : {static_cast<const void *>(a->input), static_cast<const void *>(a->selected_experts),
                          static_cast<const void *>(a->routing_weights), static_cast<const void *>(a->output)}) {
        if (reinterpret_cast<uintptr_t>(p) % alignof(float) != 0) {
            return 0;
        }
    }
    const uint64_t block_bytes = a->weight_type == GGML_TYPE_Q6_K ? sizeof(block_q6_K) : sizeof(block_q4_K);
    const uint64_t weight_alignment = a->weight_type == GGML_TYPE_Q6_K ? alignof(uint16_t) : alignof(uint32_t);
    const uint64_t row_bytes = uint64_t(a->input_size / QK_K) * block_bytes;
    if (reinterpret_cast<uintptr_t>(a->down) % weight_alignment != 0 ||
        a->down_row_stride < row_bytes || a->down_row_stride % weight_alignment != 0 ||
        a->down_row_stride > uint64_t(PTRDIFF_MAX) / a->output_rows ||
        a->down_expert_stride % weight_alignment != 0 ||
        a->down_expert_stride < (a->output_rows - 1) * a->down_row_stride + row_bytes ||
        a->down_expert_stride > uint64_t(PTRDIFF_MAX) / a->expert_count ||
        a->input_expert_stride < uint64_t(a->input_size) * sizeof(float) || a->input_expert_stride % sizeof(float) != 0 ||
        a->input_expert_stride > uint64_t(PTRDIFF_MAX) / a->selected_count ||
        a->routing_weight_stride < sizeof(float) || a->routing_weight_stride % sizeof(float) != 0 ||
        a->routing_weight_stride > uint64_t(PTRDIFF_MAX) / a->selected_count) {
        return 0;
    }
    return 1;
}

extern "C" hipError_t qwen3moe_down_launch(const qwen3moe_down_args * a, hipStream_t stream) {
    if (!qwen3moe_down_supported(a)) {
        return hipErrorInvalidValue;
    }
    return a->weight_type == GGML_TYPE_Q6_K ? dispatch<true>(*a, stream) : dispatch<false>(*a, stream);
}
