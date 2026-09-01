#include "gptoss-decode-aql.h"

#include "extensions/hip-aql.h"
#include "gptoss-config.h"
#include "gptoss-kernel-hip.h"
#include "llama-impl.h"

#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

namespace {

struct alignas(64) gptoss_decode_grid_sync {
    uint64_t multigrid_sync;
    uint32_t grid_id;
    uint32_t num_grids;
    uint64_t prev_sum;
    uint64_t all_sum;
    uint32_t single_grid_sync[2];
    uint32_t num_workgroups;
};

// This layout matches the AMDHSA v5 and v6 implicit arguments.
struct alignas(16) gptoss_decode_kernarg {
    gptoss_decode_layer_params params;
    uint32_t                   block_count[3];
    uint16_t                   group_size[3];
    uint16_t                   remainder[3];
    std::byte                  reserved_0[16];
    uint64_t                   global_offset[3];
    uint16_t                   grid_dims;
    std::byte                  reserved_1[22];
    gptoss_decode_grid_sync *  grid_sync;
    std::byte                  reserved_2[160];
};

static_assert(sizeof(gptoss_decode_grid_sync) == 64);
static_assert(offsetof(gptoss_decode_grid_sync, num_workgroups) == 40);
static_assert(offsetof(gptoss_decode_kernarg, block_count) == 272);
static_assert(offsetof(gptoss_decode_kernarg, group_size) == 284);
static_assert(offsetof(gptoss_decode_kernarg, grid_dims) == 336);
static_assert(offsetof(gptoss_decode_kernarg, grid_sync) == 360);
static_assert(sizeof(gptoss_decode_kernarg) == 528);

uint16_t gptoss_packet_header(hsa_fence_scope_t acquire, hsa_fence_scope_t release) {
    return static_cast<uint16_t>(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1u << HSA_PACKET_HEADER_BARRIER) |
        (acquire << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (release << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

bool gptoss_kernel_compatible(const llama_hip_aql_kernel & kernel) {
    return kernel.kernarg_segment_size == sizeof(gptoss_decode_kernarg) &&
           kernel.kernarg_segment_alignment <= alignof(gptoss_decode_kernarg) && !kernel.dynamic_callstack;
}

}  // namespace

struct gptoss_decode_aql {
    llama_hip_aql_queue *    queue     = nullptr;
    gptoss_decode_kernarg *   kernargs  = nullptr;
    gptoss_decode_grid_sync * grid_sync = nullptr;
    std::array<hsa_kernel_dispatch_packet_t, gptoss_layer_count> packets{};
};

gptoss_decode_aql * gptoss_decode_aql_create(int multiprocessor_count) {
    void * marker_address = nullptr;
    int    active_blocks  = 0;
    const hipError_t hip_status = gptoss_decode_aql_get_launch_info(&marker_address, &active_blocks);
    if (hip_status != hipSuccess) {
        LLAMA_LOG_ERROR("gptoss-aql: decode launch info query failed: %s\n", hipGetErrorString(hip_status));
        return nullptr;
    }
    if (active_blocks * multiprocessor_count < static_cast<int>(gptoss_decode_grid_blocks)) {
        LLAMA_LOG_ERROR("gptoss-aql: decode grid exceeds cooperative occupancy\n");
        return nullptr;
    }

    auto * aql = new (std::nothrow) gptoss_decode_aql;
    if (aql == nullptr) {
        return nullptr;
    }

    aql->queue = llama_hip_aql_queue_create(marker_address, gptoss_layer_count);
    if (aql->queue == nullptr) {
        gptoss_decode_aql_destroy(aql);
        return nullptr;
    }

    llama_hip_aql_kernel swa;
    llama_hip_aql_kernel full;
    if (!llama_hip_aql_queue_get_kernel(aql->queue, "gptoss_decode_layer_swa_kernel.kd", &swa) ||
        !llama_hip_aql_queue_get_kernel(aql->queue, "gptoss_decode_layer_full_kernel.kd", &full)) {
        LLAMA_LOG_ERROR("gptoss-aql: decode kernel lookup failed\n");
        gptoss_decode_aql_destroy(aql);
        return nullptr;
    }
    if (!gptoss_kernel_compatible(swa) || !gptoss_kernel_compatible(full)) {
        LLAMA_LOG_ERROR("gptoss-aql: incompatible decode kernel ABI\n");
        gptoss_decode_aql_destroy(aql);
        return nullptr;
    }
    aql->kernargs = static_cast<gptoss_decode_kernarg *>(
        llama_hip_aql_queue_alloc_kernarg(aql->queue, sizeof(gptoss_decode_kernarg) * gptoss_layer_count));
    if (aql->kernargs == nullptr ||
        hipMalloc(reinterpret_cast<void **>(&aql->grid_sync), sizeof(gptoss_decode_grid_sync)) != hipSuccess) {
        gptoss_decode_aql_destroy(aql);
        return nullptr;
    }

    std::memset(aql->kernargs, 0, sizeof(gptoss_decode_kernarg) * gptoss_layer_count);
    for (uint32_t il = 0; il < gptoss_layer_count; ++il) {
        const llama_hip_aql_kernel & kernel = il % 2 == 0 ? swa : full;
        auto & kernarg = aql->kernargs[il];
        auto & packet = aql->packets[il];
        kernarg.block_count[0] = gptoss_decode_grid_blocks;
        kernarg.block_count[1] = 1;
        kernarg.block_count[2] = 1;
        kernarg.group_size[0]  = gptoss_decode_block_x;
        kernarg.group_size[1]  = gptoss_decode_block_y;
        kernarg.group_size[2]  = 1;
        kernarg.grid_dims      = 3;
        kernarg.grid_sync      = aql->grid_sync;

        packet.header = gptoss_packet_header(
            il == 0 ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT,
            il + 1 == gptoss_layer_count ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT);
        packet.setup = static_cast<uint16_t>(3u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
        packet.workgroup_size_x = gptoss_decode_block_x;
        packet.workgroup_size_y = gptoss_decode_block_y;
        packet.workgroup_size_z = 1;
        packet.grid_size_x = gptoss_decode_grid_blocks * gptoss_decode_block_x;
        packet.grid_size_y = gptoss_decode_block_y;
        packet.grid_size_z = 1;
        packet.private_segment_size = kernel.private_segment_size;
        packet.group_segment_size   = kernel.group_segment_size;
        packet.kernel_object        = kernel.object;
        packet.kernarg_address      = &aql->kernargs[il];
    }

    gptoss_decode_grid_sync grid_sync{};
    grid_sync.num_workgroups = gptoss_decode_grid_blocks;
    if (hipMemcpy(aql->grid_sync, &grid_sync, sizeof(grid_sync), hipMemcpyHostToDevice) != hipSuccess) {
        gptoss_decode_aql_destroy(aql);
        return nullptr;
    }

    return aql;
}

void gptoss_decode_aql_destroy(gptoss_decode_aql * aql) {
    if (aql == nullptr) {
        return;
    }

    llama_hip_aql_free_kernarg(aql->kernargs);
    if (aql->grid_sync != nullptr) {
        (void) hipFree(aql->grid_sync);
    }
    llama_hip_aql_queue_destroy(aql->queue);
    delete aql;
}

bool gptoss_decode_aql_launch(
        gptoss_decode_aql * aql,
        const gptoss_decode_layer_params * params) {
    if (aql == nullptr || params == nullptr) {
        return false;
    }

    for (uint32_t il = 0; il < gptoss_layer_count; ++il) {
        aql->kernargs[il].params = params[il];
    }

    return llama_hip_aql_queue_submit_and_wait(aql->queue, aql->packets.data(), aql->packets.size());
}
