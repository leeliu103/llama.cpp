#include "gptoss-repack-hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace {

constexpr uint32_t repack_threads = 256;

constexpr uint32_t q8_0_scale_bytes  = sizeof(uint16_t);
constexpr uint32_t q8_0_block_values = 32;
constexpr uint32_t q8_0_block_bytes  = q8_0_scale_bytes + q8_0_block_values;

constexpr uint32_t hidden_size    = 2880;
constexpr uint32_t query_size     = 4096;
constexpr uint32_t key_value_size = 512;

constexpr size_t q8_0_query_source_size =
    static_cast<size_t>(query_size) * hidden_size / q8_0_block_values * q8_0_block_bytes;
constexpr size_t q8_0_key_value_source_size =
    static_cast<size_t>(key_value_size) * hidden_size / q8_0_block_values * q8_0_block_bytes;
constexpr size_t q8_0_qkv_value_count              = static_cast<size_t>(query_size + 2 * key_value_size) * hidden_size;
constexpr size_t q8_0_attention_output_value_count = static_cast<size_t>(hidden_size) * query_size;
constexpr size_t q8_0_attention_output_source_size =
    q8_0_attention_output_value_count / q8_0_block_values * q8_0_block_bytes;

constexpr uint32_t mxfp4_scale_bytes       = sizeof(uint8_t);
constexpr uint32_t mxfp4_block_values      = 32;
constexpr uint32_t mxfp4_block_bytes       = mxfp4_scale_bytes + mxfp4_block_values / 2;
constexpr uint32_t ogs_alignment           = 256;
constexpr uint32_t expert_dimension        = 2880;
constexpr uint32_t padded_expert_dimension = (expert_dimension + ogs_alignment - 1) / ogs_alignment * ogs_alignment;
constexpr uint32_t expert_count            = 32;

constexpr uint32_t mxfp4_source_row_size       = expert_dimension / mxfp4_block_values * mxfp4_block_bytes;
constexpr uint32_t mxfp4_source_value_row_size = expert_dimension / 2;
constexpr uint32_t mxfp4_source_scale_row_size = expert_dimension / mxfp4_block_values;
constexpr uint32_t mxfp4_packed_value_row_size = padded_expert_dimension / 2;
constexpr uint32_t mxfp4_packed_scale_row_size = padded_expert_dimension / mxfp4_block_values;

constexpr size_t mxfp4_native_weight_size =
    static_cast<size_t>(expert_count) * expert_dimension * mxfp4_source_row_size;
constexpr size_t mxfp4_down_values_size =
    static_cast<size_t>(expert_count) * padded_expert_dimension * mxfp4_packed_value_row_size;
constexpr size_t mxfp4_down_scales_size =
    static_cast<size_t>(expert_count) * padded_expert_dimension * mxfp4_packed_scale_row_size;
constexpr size_t mxfp4_gate_up_values_size   = 2 * mxfp4_down_values_size;
constexpr size_t mxfp4_gate_up_values_offset = mxfp4_down_values_size + mxfp4_down_scales_size;
constexpr size_t mxfp4_gate_up_scales_offset = mxfp4_gate_up_values_offset + mxfp4_gate_up_values_size;

constexpr uint32_t expert_bias_elements = expert_count * expert_dimension;
constexpr size_t   expert_bias_size     = static_cast<size_t>(expert_bias_elements) * sizeof(float);

__global__ void repack_q8_0_kernel(int8_t *        dst_values,
                                   uint16_t *      dst_scale_bits,
                                   const uint8_t * src,
                                   uint64_t        value_count) {
    const uint64_t value_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (value_index >= value_count) {
        return;
    }

    const uint64_t  block_index    = value_index / q8_0_block_values;
    const uint32_t  value_in_block = value_index % q8_0_block_values;
    const uint8_t * block          = src + block_index * q8_0_block_bytes;

    dst_values[value_index] = static_cast<int8_t>(block[q8_0_scale_bytes + value_in_block]);

    if (value_in_block == 0) {
        dst_scale_bits[block_index] = static_cast<uint16_t>(block[0]) | static_cast<uint16_t>(block[1]) << 8;
    }
}

__device__ uint8_t repack_mxfp4_pair(const uint8_t * block, uint32_t pair) {
    const uint32_t k0 = 2 * pair;
    const uint32_t k1 = k0 + 1;

    const uint8_t q0_byte = block[mxfp4_scale_bytes + (k0 & 15)];
    const uint8_t q1_byte = block[mxfp4_scale_bytes + (k1 & 15)];
    const uint8_t q0      = k0 < 16 ? q0_byte & 0x0f : q0_byte >> 4;
    const uint8_t q1      = k1 < 16 ? q1_byte & 0x0f : q1_byte >> 4;

    return q0 | q1 << 4;
}

__global__ void repack_mxfp4_down_kernel(uint8_t * dst_values, uint8_t * dst_scales, const uint8_t * down) {
    const uint32_t k_byte      = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t logical_row = blockIdx.y;
    const uint32_t expert      = blockIdx.z;
    const size_t   dst_row     = static_cast<size_t>(expert) * padded_expert_dimension + logical_row;

    const uint8_t * src_row =
        logical_row < expert_dimension ?
            down + (static_cast<size_t>(expert) * expert_dimension + logical_row) * mxfp4_source_row_size :
            nullptr;

    if (k_byte < mxfp4_packed_value_row_size) {
        uint8_t value = 0;

        if (src_row != nullptr && k_byte < mxfp4_source_value_row_size) {
            const uint8_t * block = src_row + (k_byte / 16) * mxfp4_block_bytes;
            value                 = repack_mxfp4_pair(block, k_byte % 16);
        }

        dst_values[dst_row * mxfp4_packed_value_row_size + k_byte] = value;
    }

    if (k_byte < mxfp4_packed_scale_row_size) {
        uint8_t scale = 0;

        if (src_row != nullptr && k_byte < mxfp4_source_scale_row_size) {
            scale = src_row[k_byte * mxfp4_block_bytes];
        }

        dst_scales[dst_row * mxfp4_packed_scale_row_size + k_byte] = scale;
    }
}

__global__ void repack_mxfp4_gate_up_kernel(uint8_t *       dst_values,
                                            uint8_t *       dst_scales,
                                            const uint8_t * gate,
                                            const uint8_t * up) {
    const uint32_t  k_byte       = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t  physical_row = blockIdx.y;
    const uint32_t  logical_row  = physical_row / 2;
    const uint32_t  expert       = blockIdx.z;
    const uint8_t * source       = physical_row % 2 == 0 ? gate : up;
    const size_t    dst_row      = static_cast<size_t>(expert) * (2 * padded_expert_dimension) + physical_row;

    const uint8_t * src_row =
        logical_row < expert_dimension ?
            source + (static_cast<size_t>(expert) * expert_dimension + logical_row) * mxfp4_source_row_size :
            nullptr;

    if (k_byte < mxfp4_packed_value_row_size) {
        uint8_t value = 0;

        if (src_row != nullptr && k_byte < mxfp4_source_value_row_size) {
            const uint8_t * block = src_row + (k_byte / 16) * mxfp4_block_bytes;
            value                 = repack_mxfp4_pair(block, k_byte % 16);
        }

        dst_values[dst_row * mxfp4_packed_value_row_size + k_byte] = value;
    }

    if (k_byte < mxfp4_packed_scale_row_size) {
        uint8_t scale = 0;

        if (src_row != nullptr && k_byte < mxfp4_source_scale_row_size) {
            scale = src_row[k_byte * mxfp4_block_bytes];
        }

        dst_scales[dst_row * mxfp4_packed_scale_row_size + k_byte] = scale;
    }
}

__global__ void repack_gate_up_bias_kernel(float * dst, const float * gate, const float * up) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= expert_bias_elements) {
        return;
    }

    dst[2 * index]     = gate[index];
    dst[2 * index + 1] = up[index];
}

}  // namespace

bool gptoss_repack_qkv_launch(uint8_t * q, uint8_t * k, uint8_t * v, uint8_t * scratch) {
    if (hipMemcpyAsync(scratch, q, q8_0_query_source_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess ||
        hipMemcpyAsync(scratch + q8_0_query_source_size, k, q8_0_key_value_source_size, hipMemcpyDeviceToDevice,
                       nullptr) != hipSuccess ||
        hipMemcpyAsync(scratch + q8_0_query_source_size + q8_0_key_value_source_size, v, q8_0_key_value_source_size,
                       hipMemcpyDeviceToDevice, nullptr) != hipSuccess) {
        return false;
    }

    int8_t *       packed_values     = reinterpret_cast<int8_t *>(q);
    uint16_t *     packed_scale_bits = reinterpret_cast<uint16_t *>(q + q8_0_qkv_value_count);
    const uint32_t grid_size = static_cast<uint32_t>((q8_0_qkv_value_count + repack_threads - 1) / repack_threads);

    hipLaunchKernelGGL(repack_q8_0_kernel, dim3(grid_size), dim3(repack_threads), 0, nullptr, packed_values,
                       packed_scale_bits, scratch, static_cast<uint64_t>(q8_0_qkv_value_count));

    return hipGetLastError() == hipSuccess;
}

bool gptoss_repack_attention_output_launch(uint8_t * weight, uint8_t * scratch) {
    if (hipMemcpyAsync(scratch, weight, q8_0_attention_output_source_size, hipMemcpyDeviceToDevice, nullptr) !=
        hipSuccess) {
        return false;
    }

    int8_t *       packed_values     = reinterpret_cast<int8_t *>(weight);
    uint16_t *     packed_scale_bits = reinterpret_cast<uint16_t *>(weight + q8_0_attention_output_value_count);
    const uint32_t grid_size =
        static_cast<uint32_t>((q8_0_attention_output_value_count + repack_threads - 1) / repack_threads);

    hipLaunchKernelGGL(repack_q8_0_kernel, dim3(grid_size), dim3(repack_threads), 0, nullptr, packed_values,
                       packed_scale_bits, scratch, static_cast<uint64_t>(q8_0_attention_output_value_count));

    return hipGetLastError() == hipSuccess;
}

bool gptoss_repack_moe_launch(uint8_t * gate,
                              uint8_t * down,
                              uint8_t * up,
                              float *   gate_bias,
                              float *   down_bias,
                              float *   up_bias,
                              uint8_t * scratch) {
    uint8_t * saved_gate = scratch;
    uint8_t * saved_down = saved_gate + mxfp4_native_weight_size;
    uint8_t * saved_up   = saved_down + mxfp4_native_weight_size;

    if (hipMemcpyAsync(saved_gate, gate, mxfp4_native_weight_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess ||
        hipMemcpyAsync(saved_down, down, mxfp4_native_weight_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess ||
        hipMemcpyAsync(saved_up, up, mxfp4_native_weight_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess) {
        return false;
    }

    uint8_t * packed_down_values    = gate;
    uint8_t * packed_down_scales    = gate + mxfp4_down_values_size;
    uint8_t * packed_gate_up_values = gate + mxfp4_gate_up_values_offset;
    uint8_t * packed_gate_up_scales = gate + mxfp4_gate_up_scales_offset;

    const uint32_t blocks_per_row = (mxfp4_packed_value_row_size + repack_threads - 1) / repack_threads;

    hipLaunchKernelGGL(repack_mxfp4_down_kernel, dim3(blocks_per_row, padded_expert_dimension, expert_count),
                       dim3(repack_threads), 0, nullptr, packed_down_values, packed_down_scales, saved_down);

    if (hipGetLastError() != hipSuccess) {
        return false;
    }

    hipLaunchKernelGGL(repack_mxfp4_gate_up_kernel, dim3(blocks_per_row, 2 * padded_expert_dimension, expert_count),
                       dim3(repack_threads), 0, nullptr, packed_gate_up_values, packed_gate_up_scales, saved_gate,
                       saved_up);

    if (hipGetLastError() != hipSuccess) {
        return false;
    }

    float * saved_gate_bias     = reinterpret_cast<float *>(scratch);
    float * saved_up_bias       = saved_gate_bias + expert_bias_elements;
    float * packed_down_bias    = gate_bias;
    float * packed_gate_up_bias = down_bias;

    if (hipMemcpyAsync(saved_gate_bias, gate_bias, expert_bias_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess ||
        hipMemcpyAsync(saved_up_bias, up_bias, expert_bias_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess ||
        hipMemcpyAsync(packed_down_bias, down_bias, expert_bias_size, hipMemcpyDeviceToDevice, nullptr) != hipSuccess) {
        return false;
    }

    const uint32_t bias_grid_size = (expert_bias_elements + repack_threads - 1) / repack_threads;

    hipLaunchKernelGGL(repack_gate_up_bias_kernel, dim3(bias_grid_size), dim3(repack_threads), 0, nullptr,
                       packed_gate_up_bias, saved_gate_bias, saved_up_bias);

    return hipGetLastError() == hipSuccess;
}
