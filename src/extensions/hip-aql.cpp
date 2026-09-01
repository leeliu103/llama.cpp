#include "hip-aql.h"

#include "llama-impl.h"

#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {

bool llama_hip_aql_hsa_ok(hsa_status_t status, const char * operation) {
    if (status == HSA_STATUS_SUCCESS) {
        return true;
    }

    const char * message = nullptr;
    (void) hsa_status_string(status, &message);
    LLAMA_LOG_ERROR("hip-aql: %s failed: %s (%d)\n", operation, message != nullptr ? message : "unknown error",
                    static_cast<int>(status));
    return false;
}

struct llama_hip_aql_pool_search {
    hsa_agent_t           agent{};
    hsa_amd_memory_pool_t pool{};
};

hsa_status_t llama_hip_aql_find_pool(hsa_amd_memory_pool_t pool, void * data) {
    auto * search = static_cast<llama_hip_aql_pool_search *>(data);

    hsa_amd_segment_t segment{};
    hsa_status_t status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (status != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL) {
        return status;
    }

    uint32_t flags = 0;
    bool     allocatable = false;
    status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allocatable);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }

    hsa_amd_memory_pool_access_t access{};
    status = hsa_amd_agent_memory_pool_get_info(
        search->agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    if (!allocatable || (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) == 0 ||
        access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
        return HSA_STATUS_SUCCESS;
    }

    search->pool = pool;
    return HSA_STATUS_INFO_BREAK;
}

hsa_status_t llama_hip_aql_find_pool_agent(hsa_agent_t agent, void * data) {
    auto * search = static_cast<llama_hip_aql_pool_search *>(data);
    const hsa_status_t status = hsa_amd_agent_iterate_memory_pools(agent, llama_hip_aql_find_pool, search);
    if (status == HSA_STATUS_INFO_BREAK) {
        return HSA_STATUS_INFO_BREAK;
    }
    return status;
}

uint32_t llama_hip_aql_next_power_of_two(uint32_t value) {
    uint32_t result = 1;
    while (result < value && result <= std::numeric_limits<uint32_t>::max() / 2) {
        result <<= 1;
    }
    return result;
}

bool llama_hip_aql_find_code_object(
        const void * marker_address,
        hsa_agent_t * agent,
        hsa_executable_t * executable) {
    uint16_t minor_version = 0;
    bool     supported     = false;
    if (!llama_hip_aql_hsa_ok(
            hsa_system_major_extension_supported(HSA_EXTENSION_AMD_LOADER, 1, &minor_version, &supported),
            "loader extension query") ||
        !supported) {
        LLAMA_LOG_ERROR("hip-aql: AMD loader extension is unavailable\n");
        return false;
    }

    hsa_ven_amd_loader_1_00_pfn_t loader{};
    if (!llama_hip_aql_hsa_ok(
            hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader),
            "loader extension table query") ||
        loader.hsa_ven_amd_loader_query_segment_descriptors == nullptr) {
        return false;
    }

    size_t count = 0;
    if (!llama_hip_aql_hsa_ok(
            loader.hsa_ven_amd_loader_query_segment_descriptors(nullptr, &count), "segment count query") ||
        count == 0) {
        return false;
    }

    std::vector<hsa_ven_amd_loader_segment_descriptor_t> descriptors(count);
    if (!llama_hip_aql_hsa_ok(
            loader.hsa_ven_amd_loader_query_segment_descriptors(descriptors.data(), &count),
            "segment descriptor query")) {
        return false;
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(marker_address);
    for (const auto & descriptor : descriptors) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(descriptor.segment_base);
        if (address >= base && address - base < descriptor.segment_size && descriptor.agent.handle != 0 &&
            descriptor.executable.handle != 0) {
            *agent      = descriptor.agent;
            *executable = descriptor.executable;
            return true;
        }
    }

    LLAMA_LOG_ERROR("hip-aql: executable segment lookup failed\n");
    return false;
}

uint32_t llama_hip_aql_packet_full_header(const hsa_kernel_dispatch_packet_t & packet) {
    uint32_t result;
    std::memcpy(&result, &packet, sizeof(result));
    return result;
}

static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64);

}  // namespace

struct llama_hip_aql_queue {
    hsa_agent_t           agent{};
    hsa_executable_t      executable{};
    hsa_amd_memory_pool_t kernarg_pool{};
    hsa_queue_t *         queue = nullptr;
    hsa_signal_t          completion{};
};

llama_hip_aql_queue * llama_hip_aql_queue_create(
        const void * marker_address,
        uint32_t minimum_packet_count) {
    if (marker_address == nullptr) {
        return nullptr;
    }

    auto * result = new (std::nothrow) llama_hip_aql_queue;
    if (result == nullptr) {
        return nullptr;
    }

    if (!llama_hip_aql_hsa_ok(hsa_init(), "runtime initialization")) {
        delete result;
        return nullptr;
    }
    if (!llama_hip_aql_find_code_object(marker_address, &result->agent, &result->executable)) {
        llama_hip_aql_queue_destroy(result);
        return nullptr;
    }

    llama_hip_aql_pool_search pool_search{result->agent};
    hsa_status_t status = hsa_iterate_agents(llama_hip_aql_find_pool_agent, &pool_search);
    if ((status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) || pool_search.pool.handle == 0) {
        (void) llama_hip_aql_hsa_ok(status, "kernarg pool enumeration");
        LLAMA_LOG_ERROR("hip-aql: no allocatable kernarg pool found\n");
        llama_hip_aql_queue_destroy(result);
        return nullptr;
    }
    result->kernarg_pool = pool_search.pool;

    const uint32_t queue_size = llama_hip_aql_next_power_of_two(minimum_packet_count);
    if (queue_size < minimum_packet_count) {
        LLAMA_LOG_ERROR("hip-aql: requested packet batch does not fit the cooperative queue\n");
        llama_hip_aql_queue_destroy(result);
        return nullptr;
    }

    if (!llama_hip_aql_hsa_ok(
            hsa_queue_create(result->agent, queue_size, HSA_QUEUE_TYPE_COOPERATIVE, nullptr, nullptr,
                             std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(),
                             &result->queue),
            "cooperative queue creation") ||
        result->queue == nullptr || result->queue->base_address == nullptr ||
        (result->queue->features & HSA_QUEUE_FEATURE_KERNEL_DISPATCH) == 0 ||
        result->queue->size < minimum_packet_count || (result->queue->size & (result->queue->size - 1)) != 0) {
        LLAMA_LOG_ERROR("hip-aql: invalid cooperative queue\n");
        llama_hip_aql_queue_destroy(result);
        return nullptr;
    }
    if (!llama_hip_aql_hsa_ok(hsa_signal_create(0, 0, nullptr, &result->completion), "signal creation")) {
        llama_hip_aql_queue_destroy(result);
        return nullptr;
    }
    return result;
}

void llama_hip_aql_queue_destroy(llama_hip_aql_queue * queue) {
    if (queue == nullptr) {
        return;
    }

    if (queue->queue != nullptr) {
        (void) llama_hip_aql_hsa_ok(hsa_queue_destroy(queue->queue), "queue destruction");
    }
    if (queue->completion.handle != 0) {
        (void) llama_hip_aql_hsa_ok(hsa_signal_destroy(queue->completion), "signal destruction");
    }
    (void) llama_hip_aql_hsa_ok(hsa_shut_down(), "runtime shutdown");
    delete queue;
}

bool llama_hip_aql_queue_get_kernel(
        const llama_hip_aql_queue * queue,
        const char * symbol_name,
        llama_hip_aql_kernel * kernel) {
    if (queue == nullptr || symbol_name == nullptr || kernel == nullptr) {
        return false;
    }

    hsa_executable_symbol_t symbol{};
    hsa_symbol_kind_t       kind{};

    if (!llama_hip_aql_hsa_ok(hsa_executable_get_symbol_by_name(
            queue->executable, symbol_name, &queue->agent, &symbol), "kernel lookup") ||
        !llama_hip_aql_hsa_ok(
            hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind), "kernel type query") ||
        kind != HSA_SYMBOL_KIND_KERNEL) {
        return false;
    }

    return llama_hip_aql_hsa_ok(
               hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel->object),
               "kernel object query") &&
           llama_hip_aql_hsa_ok(
               hsa_executable_symbol_get_info(
                   symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernel->kernarg_segment_size),
               "kernarg size query") &&
           llama_hip_aql_hsa_ok(
               hsa_executable_symbol_get_info(
                   symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                   &kernel->kernarg_segment_alignment),
               "kernarg alignment query") &&
           llama_hip_aql_hsa_ok(
               hsa_executable_symbol_get_info(
                   symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kernel->group_segment_size),
               "group segment query") &&
           llama_hip_aql_hsa_ok(
               hsa_executable_symbol_get_info(
                   symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kernel->private_segment_size),
               "private segment query") &&
           llama_hip_aql_hsa_ok(
               hsa_executable_symbol_get_info(
                   symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &kernel->dynamic_callstack),
               "dynamic callstack query") &&
           kernel->object != 0;
}

void * llama_hip_aql_queue_alloc_kernarg(llama_hip_aql_queue * queue, size_t size) {
    if (queue == nullptr || size == 0) {
        return nullptr;
    }

    void * pointer = nullptr;
    if (!llama_hip_aql_hsa_ok(
            hsa_amd_memory_pool_allocate(
                queue->kernarg_pool, size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &pointer),
            "kernarg memory allocation")) {
        return nullptr;
    }
    if (!llama_hip_aql_hsa_ok(
            hsa_amd_agents_allow_access(1, &queue->agent, nullptr, pointer), "kernarg memory access")) {
        (void) hsa_amd_memory_pool_free(pointer);
        return nullptr;
    }
    return pointer;
}

void llama_hip_aql_free_kernarg(void * pointer) {
    if (pointer != nullptr) {
        (void) llama_hip_aql_hsa_ok(hsa_amd_memory_pool_free(pointer), "kernarg memory free");
    }
}

bool llama_hip_aql_queue_submit_and_wait(
        llama_hip_aql_queue * queue,
        const hsa_kernel_dispatch_packet_t * packets,
        size_t count) {
    if (queue == nullptr || packets == nullptr || count == 0 || count > queue->queue->size) {
        return false;
    }

    hsa_signal_store_screlease(queue->completion, 1);

    const uint64_t start = hsa_queue_add_write_index_scacq_screl(queue->queue, count);
    const uint64_t end   = start + count - 1;
    while (end - hsa_queue_load_read_index_scacquire(queue->queue) >= queue->queue->size) {
        std::this_thread::yield();
    }

    auto * base = static_cast<hsa_kernel_dispatch_packet_t *>(queue->queue->base_address);
    const uint32_t invalid_header = static_cast<uint32_t>(HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);
    auto * first = &base[start & (queue->queue->size - 1)];
    __atomic_store_n(reinterpret_cast<uint32_t *>(first), invalid_header, __ATOMIC_RELAXED);
    std::memcpy(reinterpret_cast<std::byte *>(first) + sizeof(uint32_t),
                reinterpret_cast<const std::byte *>(&packets[0]) + sizeof(uint32_t),
                sizeof(*first) - sizeof(uint32_t));

    for (size_t i = 1; i < count; ++i) {
        auto * destination = &base[(start + i) & (queue->queue->size - 1)];
        std::memcpy(destination, &packets[i], sizeof(*destination));
    }
    auto * last = &base[end & (queue->queue->size - 1)];
    last->completion_signal = queue->completion;

    __atomic_store_n(reinterpret_cast<uint32_t *>(first), llama_hip_aql_packet_full_header(packets[0]),
                     __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue->queue->doorbell_signal, static_cast<hsa_signal_value_t>(end));

    hsa_signal_value_t completion;
    do {
        completion = hsa_signal_wait_scacquire(queue->completion, HSA_SIGNAL_CONDITION_LT, 1,
                                               std::numeric_limits<uint64_t>::max(), HSA_WAIT_STATE_ACTIVE);
    } while (completion >= 1);
    return completion == 0;
}
