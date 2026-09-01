#pragma once

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>

struct llama_hip_aql_queue;

struct llama_hip_aql_kernel {
    uint64_t object                     = 0;
    uint32_t kernarg_segment_size       = 0;
    uint32_t kernarg_segment_alignment  = 0;
    uint32_t group_segment_size         = 0;
    uint32_t private_segment_size       = 0;
    bool     dynamic_callstack          = false;
};

llama_hip_aql_queue * llama_hip_aql_queue_create(
    const void * marker_address,
    uint32_t minimum_packet_count);

void llama_hip_aql_queue_destroy(llama_hip_aql_queue * queue);

bool llama_hip_aql_queue_get_kernel(
    const llama_hip_aql_queue * queue,
    const char * symbol_name,
    llama_hip_aql_kernel * kernel);

void * llama_hip_aql_queue_alloc_kernarg(llama_hip_aql_queue * queue, size_t size);

void llama_hip_aql_free_kernarg(void * pointer);

bool llama_hip_aql_queue_submit_and_wait(
    llama_hip_aql_queue * queue,
    const hsa_kernel_dispatch_packet_t * packets,
    size_t count);
