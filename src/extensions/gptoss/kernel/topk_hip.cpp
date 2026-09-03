#include "../gptoss-config.h"

#include <float.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int expert_used_count = gptoss_expert_used_count;
constexpr int warp_size         = 32;
constexpr int experts_per_lane  = gptoss_max_expert_count / warp_size;

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
    uint32_t n_tokens,
    uint32_t expert_count) {
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tokens) {
        return;
    }

    const int lane = threadIdx.x;
    float     scores[experts_per_lane];

#pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        const int expert = lane + i * warp_size;
        float     score  = -INFINITY;
        if (expert < (int) expert_count) {
            score = router_logits[token * expert_count + expert] + router_bias[expert];
        }
        scores[i] = isnan(score) ? -FLT_MAX : score;
    }

    float selected_logit = -INFINITY;

#pragma unroll
    for (int selected = 0; selected < expert_used_count; ++selected) {
        float best_logit  = scores[0];
        int   best_expert = lane;

#pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * warp_size;
            if (scores[i] > best_logit || (scores[i] == best_logit && expert < best_expert)) {
                best_logit  = scores[i];
                best_expert = expert;
            }
        }

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
            selected_ids[token * expert_used_count + selected] = best_expert;
            selected_logit                                     = best_logit;
        }

#pragma unroll
        for (int i = 0; i < experts_per_lane; ++i) {
            if (lane + i * warp_size == best_expert) {
                scores[i] = -INFINITY;
            }
        }
    }

    const float max_logit  = warp_max(selected_logit);
    float       weight     = lane < expert_used_count ? expf(selected_logit - max_logit) : 0.0f;
    const float weight_sum = warp_sum(weight);

    if (lane < expert_used_count) {
        selected_weights[token * expert_used_count + lane] = weight / weight_sum;
    }
}
