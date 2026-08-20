#include "../gptoss-config.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t gptoss_routing_threads = 256;

}  // namespace

__launch_bounds__(gptoss_routing_threads, 1) __global__ void gptoss_ogs_build_routes(
    const int32_t * __restrict__ expert_ids,
    int32_t * __restrict__ gather_token_indices,
    int32_t * __restrict__ scatter_route_indices,
    int32_t * __restrict__ expert_counts,
    int32_t * __restrict__ route_offsets,
    int32_t * __restrict__ block_offsets,
    int32_t * __restrict__ block_schedule,
    uint32_t route_count,
    uint32_t schedule_capacity) {
    __shared__ int32_t histogram[gptoss_expert_count];
    __shared__ int32_t cursors[gptoss_expert_count];
    __shared__ int32_t tile_offsets[gptoss_expert_count];

    const uint32_t thread = threadIdx.x;
    if (thread < gptoss_expert_count) {
        histogram[thread] = 0;
    }
    for (uint32_t index = thread; index < schedule_capacity; index += blockDim.x) {
        block_schedule[index] = -1;
    }
    __syncthreads();

    for (uint32_t route = thread; route < route_count; route += blockDim.x) {
        const int32_t expert = expert_ids[route];
        if ((uint32_t) expert < gptoss_expert_count) {
            atomicAdd(&histogram[expert], 1);
        }
    }
    __syncthreads();

    if (thread < gptoss_expert_count) {
        expert_counts[thread] = histogram[thread];
    }
    if (thread == 0) {
        int32_t route_offset = 0;
        int32_t tile_offset  = 0;
        for (uint32_t expert = 0; expert < gptoss_expert_count; ++expert) {
            route_offsets[expert] = route_offset;
            block_offsets[expert] = tile_offset;
            cursors[expert]       = route_offset;
            tile_offsets[expert]  = tile_offset;
            route_offset += histogram[expert];
            tile_offset += (histogram[expert] + (int32_t) gptoss_ogs_block_m - 1) / (int32_t) gptoss_ogs_block_m;
        }
        route_offsets[gptoss_expert_count] = route_offset;
        block_offsets[gptoss_expert_count] = tile_offset;
    }
    __syncthreads();

    for (uint32_t route = thread; route < route_count; route += blockDim.x) {
        const uint32_t token  = route / gptoss_expert_used_count;
        const int32_t  expert = expert_ids[route];
        if ((uint32_t) expert < gptoss_expert_count) {
            const int32_t destination          = atomicAdd(&cursors[expert], 1);
            gather_token_indices[destination]  = (int32_t) token;
            scatter_route_indices[destination] = (int32_t) route;
        }
    }

    for (uint32_t expert = thread; expert < gptoss_expert_count; expert += blockDim.x) {
        const uint32_t tile_count  = ((uint32_t) histogram[expert] + gptoss_ogs_block_m - 1) / gptoss_ogs_block_m;
        const uint32_t destination = (uint32_t) tile_offsets[expert];
        for (uint32_t tile = 0; tile < tile_count; ++tile) {
            block_schedule[destination + tile] = (int32_t) ((tile << 16) | expert);
        }
    }
}
