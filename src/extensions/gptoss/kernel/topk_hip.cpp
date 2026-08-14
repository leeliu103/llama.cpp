#include <float.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int expert_count      = 32;
constexpr int expert_used_count = 4;
constexpr int warp_size         = 32;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor(value, offset, warp_size));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, warp_size);
    }
    return value;
}

}  // namespace

__launch_bounds__(4 * warp_size) __global__ void gptoss_biased_topk_softmax_kernel(
    const float * __restrict__ router_logits,
    const float * __restrict__ router_bias,
    int32_t * __restrict__ selected_ids,
    float * __restrict__ selected_weights,
    uint32_t n_tokens) {
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tokens) {
        return;
    }

    const int lane         = threadIdx.x;
    float     biased_logit = router_logits[token * expert_count + lane] + router_bias[lane];
    if (isnan(biased_logit)) {
        biased_logit = -FLT_MAX;
    }

    float selected_logit = -INFINITY;

#pragma unroll
    for (int selected = 0; selected < expert_used_count; ++selected) {
        float best_logit  = biased_logit;
        int   best_expert = lane;

#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            const float other_logit  = __shfl_xor(best_logit, offset, warp_size);
            const int   other_expert = __shfl_xor(best_expert, offset, warp_size);
            if (other_logit > best_logit || (other_logit == best_logit && other_expert < best_expert)) {
                best_logit  = other_logit;
                best_expert = other_expert;
            }
        }

        if (lane == selected) {
            selected_ids[token * expert_count + selected] = best_expert;
            selected_logit                                = best_logit;
        }
        if (lane == best_expert) {
            biased_logit = -INFINITY;
        }
    }

    const float max_logit  = warp_max(selected_logit);
    float       weight     = lane < expert_used_count ? expf(selected_logit - max_logit) : 0.0f;
    const float weight_sum = warp_sum(weight);

    if (lane < expert_used_count) {
        selected_weights[token * expert_used_count + lane] = weight / weight_sum;
    }
}
