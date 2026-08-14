#include <hip/hip_runtime.h>

#include <cstdint>

__global__ void gptoss_moe_combine_residual_f32(float *       out,
                                                const float * residual,
                                                const float * __restrict__ expert_outputs,
                                                const float * __restrict__ routing_weights,
                                                uint32_t n_tokens,
                                                uint32_t n_experts_used,
                                                uint32_t hidden_size) {
    const uint64_t element_count = (uint64_t) n_tokens * hidden_size;
    const uint64_t index         = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    const uint32_t token       = (uint32_t) (index / hidden_size);
    const uint32_t column      = (uint32_t) (index - (uint64_t) token * hidden_size);
    const uint64_t output_base = (uint64_t) token * n_experts_used * hidden_size + column;
    const uint32_t weight_base = token * n_experts_used;

    float sum = 0.0f;
    for (uint32_t expert = 0; expert < n_experts_used; ++expert) {
        const float expert_output  = expert_outputs[output_base + (uint64_t) expert * hidden_size];
        const float routing_weight = routing_weights[weight_base + expert];
        sum                        = fmaf(expert_output, routing_weight, sum);
    }
    out[index] = residual[index] + sum;
}
