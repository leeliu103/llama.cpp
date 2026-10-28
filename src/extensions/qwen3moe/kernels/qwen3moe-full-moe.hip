#include "qwen3moe-full-moe.h"

#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr int hidden = 4096;
constexpr int intermediate = 512;
constexpr int experts = 128;
constexpr int selected = 8;
constexpr int wave_size = 32;
constexpr int threads = selected * wave_size;
constexpr int down_rows = QWEN3MOE_FULL_MOE_DOWN_ROWS;
constexpr unsigned max_barrier_polls = 1u << 20;

__device__ __forceinline__ unsigned load_agent(const uint32_t * p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void set_error(uint32_t * p, unsigned error) {
    __hip_atomic_fetch_or(p, error, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

// One interphase barrier. All workgroups participate even after invalid IDs.
// The launcher resets arrivals before every kernel, including graph replays.
// A cooperative launch and occupancy preflight are the primary residency
// checks; wall-clock and iteration bounds are independent escape mechanisms.
__device__ __forceinline__ bool arrive_and_wait(const qwen3moe_full_moe_args & a,
                                               uint64_t timeout_ticks, unsigned & proceed) {
    __threadfence();  // every producer thread publishes its H stores
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned previous = __hip_atomic_fetch_add(a.arrival_counter, 1u, __ATOMIC_ACQ_REL,
                                                         __HIP_MEMORY_SCOPE_AGENT);
        if (previous >= gridDim.x) {
            set_error(a.sticky_error, QWEN3MOE_FULL_MOE_BARRIER_COUNT);
        }
        const uint64_t start = static_cast<uint64_t>(wall_clock64());
        bool arrived = false;
        for (unsigned poll = 0; poll < max_barrier_polls; ++poll) {
            if (load_agent(a.sticky_error)) {
                break;
            }
            if (load_agent(a.arrival_counter) == gridDim.x) {
                arrived = true;
                break;
            }
            if ((poll & 63u) == 63u && static_cast<uint64_t>(wall_clock64()) - start >= timeout_ticks) {
                break;
            }
        }
        if (!arrived && !load_agent(a.sticky_error)) {
            set_error(a.sticky_error, QWEN3MOE_FULL_MOE_BARRIER_TIMEOUT);
        }
        proceed = arrived && !load_agent(a.sticky_error);
    }
    __syncthreads();
    if (!proceed) {
        return false;
    }
    // The acquire emits gfx1151 GL1/GL0 invalidation for each consumer wave.
    (void) load_agent(a.arrival_counter);
    __syncthreads();
    return true;
}

template<bool q6, int gate_rows>
__launch_bounds__(threads)
__global__ void full_moe(qwen3moe_full_moe_args a, uint64_t timeout_ticks) {
    __shared__ __align__(16) block_q8_1 q8[hidden / QK8_1];
    __shared__ int ids[selected];
    __shared__ float routing[selected];
    __shared__ unsigned proceed;
    const int lane = threadIdx.x % wave_size;
    const unsigned wave = __builtin_amdgcn_readfirstlane(threadIdx.x / wave_size);
    if (threadIdx.x < selected) {
        const int expert = a.selected_experts[threadIdx.x];
        ids[threadIdx.x] = expert;
        routing[threadIdx.x] = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(a.routing_weights) + threadIdx.x * a.routing_stride);
        if (expert < 0 || expert >= experts) {
            set_error(a.sticky_error, QWEN3MOE_FULL_MOE_INVALID_EXPERT);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        proceed = !load_agent(a.sticky_error);
    }
    __syncthreads();
    if (proceed) {
        // Preserve stock per-32 Q8_1 rounding, FP16 scale/sum and exact dots.
        for (int i = threadIdx.x; i < hidden; i += threads) {
            const float x = a.input[i];
            const float amax = warp_reduce_max<QK8_1>(fabsf(x));
            const float sum = warp_reduce_sum<QK8_1>(x);
            const float d = amax / 127.0f;
            q8[i / QK8_1].qs[lane] = amax == 0.0f ? 0 : roundf(x / d);
            if (lane == 0) q8[i / QK8_1].ds = make_half2(d, sum);
        }
        __syncthreads();
        constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
        constexpr int lanes_per_quant_block = QI4_K / vdr;
        constexpr int quant_blocks_per_iter = wave_size / lanes_per_quant_block;
        constexpr int rows_per_tile = selected * gate_rows;
        constexpr int tiles_per_expert = intermediate / rows_per_tile;
        const int kqs = vdr * (lane % lanes_per_quant_block);
        const unsigned row0 = (blockIdx.x % tiles_per_expert) * rows_per_tile + wave * gate_rows;
        const uint64_t gate_row_offset = row0 * a.gate_row_stride;
        const uint64_t up_row_offset = row0 * a.up_row_stride;
        for (unsigned slot = blockIdx.x / tiles_per_expert; slot < selected; slot += gridDim.x / tiles_per_expert) {
            const unsigned expert = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(ids[slot]));
            const char * gate = static_cast<const char *>(a.gate) + expert * a.gate_expert_stride + gate_row_offset;
            const char * up = static_cast<const char *>(a.up) + expert * a.up_expert_stride + up_row_offset;
            float g[gate_rows] = {}, u[gate_rows] = {};
            for (int k = lane / lanes_per_quant_block; k < hidden / QK_K; k += quant_blocks_per_iter) {
                const block_q8_1 * x = q8 + k * (QK_K / QK8_1);
#pragma unroll
                for (int r = 0; r < gate_rows; ++r) {
                    g[r] += vec_dot_q4_K_q8_1(gate + r * a.gate_row_stride, x, k, kqs);
                    u[r] += vec_dot_q4_K_q8_1(up + r * a.up_row_stride, x, k, kqs);
                }
            }
#pragma unroll
            for (int r = 0; r < gate_rows; ++r) {
                g[r] = warp_reduce_sum<wave_size>(g[r]);
                u[r] = warp_reduce_sum<wave_size>(u[r]);
                if (lane == 0) a.activation_scratch[slot * intermediate + row0 + r] = u[r] * ggml_cuda_op_silu_single(g[r]);
            }
        }
    }
    if (!arrive_and_wait(a, timeout_ticks, proceed)) {
        return; // uniform workgroup exit; sticky error invalidates all output
    }

    // Every wave quantizes one selected expert's 512-element activation.
    block_q8_1 * xq = q8 + wave * (intermediate / QK8_1);
    for (int i = lane; i < intermediate; i += wave_size) {
        const float x = a.activation_scratch[wave * intermediate + i];
        const float amax = warp_reduce_max<QK8_1>(fabsf(x));
        const float sum = warp_reduce_sum<QK8_1>(x);
        const float d = amax / 127.0f;
        xq[i / QK8_1].qs[lane] = amax == 0.0f ? 0 : roundf(x / d);
        if (lane == 0) xq[i / QK8_1].ds = make_half2(d, sum);
    }
    __syncthreads();
    constexpr int vdr = q6 ? VDR_Q6_K_Q8_1_MMVQ : VDR_Q4_K_Q8_1_MMVQ;
    constexpr int qi = q6 ? QI6_K : QI4_K;
    constexpr int lanes_per_quant_block = qi / vdr;
    constexpr int quant_blocks_per_iter = wave_size / lanes_per_quant_block;
    const int kqs = vdr * (lane % lanes_per_quant_block);
    constexpr int output_rows_per_tile = selected * down_rows;
    for (unsigned tile = blockIdx.x; tile < hidden / output_rows_per_tile; tile += gridDim.x) {
        const unsigned row0 = tile * output_rows_per_tile + wave * down_rows;
        const uint64_t row_offset = row0 * a.down_row_stride;
        float sums[down_rows] = {};
        for (int expert_slot = 0; expert_slot < selected; ++expert_slot) {
            const unsigned expert = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(ids[expert_slot]));
            const char * down = static_cast<const char *>(a.down) + expert * a.down_expert_stride + row_offset;
            const block_q8_1 * expert_xq = q8 + expert_slot * (intermediate / QK8_1);
            float values[down_rows] = {};
            for (int k = lane / lanes_per_quant_block; k < intermediate / QK_K; k += quant_blocks_per_iter) {
                const block_q8_1 * x = expert_xq + k * (QK_K / QK8_1);
#pragma unroll
                for (int r = 0; r < down_rows; ++r) {
                    const void * row = down + r * a.down_row_stride;
                    if constexpr (q6) values[r] += vec_dot_q6_K_q8_1(row, x, k, kqs);
                    else values[r] += vec_dot_q4_K_q8_1(row, x, k, kqs);
                }
            }
#pragma unroll
            for (int r = 0; r < down_rows; ++r) {
                values[r] = warp_reduce_sum<wave_size>(values[r]);
                if (lane == 0) {
                    float product;
                    // Preserve the separate stock multiply's F32 rounding.
                    asm("v_mul_f32 %0, %1, %2" : "=v"(product) : "v"(values[r]), "v"(routing[expert_slot]));
                    sums[r] = expert_slot == 0 ? product : sums[r] + product;
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < down_rows; ++r) a.output[row0 + r] = sums[r];
        }
    }
}

bool matrix_span(uint64_t row, uint64_t expert, uint64_t row_bytes, int rows, uint64_t alignment,
                 uint64_t & span) {
    if (row < row_bytes || row % alignment || expert % alignment ||
        row > uint64_t(PTRDIFF_MAX) / rows) return false;
    const uint64_t matrix = (rows - 1) * row + row_bytes;
    if (expert < matrix || expert > uint64_t(PTRDIFF_MAX) / experts) return false;
    span = (experts - 1) * expert + matrix;
    return true;
}
struct memory_span { const void * pointer; uint64_t bytes; };
bool overlaps(memory_span a, memory_span b) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a.pointer), y = reinterpret_cast<uintptr_t>(b.pointer);
    return x < y + b.bytes && y < x + a.bytes;
}

template<bool q6, int gate_rows>
hipError_t occupancy(int * active) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(active, full_moe<q6, gate_rows>, threads, 0);
}
template<bool q6, int gate_rows>
hipError_t submit(const qwen3moe_full_moe_args & a, uint64_t timeout, hipStream_t stream) {
    auto copy = a;
    void * parameters[] = {&copy, &timeout};
    return hipLaunchCooperativeKernel(reinterpret_cast<const void *>(full_moe<q6, gate_rows>),
                                     dim3(a.grid_blocks), dim3(threads), parameters, 0, stream);
}
} // namespace

extern "C" int qwen3moe_full_moe_supported(const qwen3moe_full_moe_args * a) {
    if (!a || !a->gate || !a->up || !a->down || !a->input || !a->selected_experts || !a->routing_weights ||
        !a->output || !a->activation_scratch || !a->arrival_counter || !a->sticky_error ||
        (a->down_type != GGML_TYPE_Q4_K && a->down_type != GGML_TYPE_Q6_K) ||
        (a->grid_blocks != 32 && a->grid_blocks != 64 && a->grid_blocks != 128) ||
        (a->gate_rows_per_wave != 2 && a->gate_rows_per_wave != 4)) return 0;
    for (const void * p : {a->gate, a->up, a->down, static_cast<const void *>(a->input),
         static_cast<const void *>(a->selected_experts), static_cast<const void *>(a->routing_weights),
         static_cast<const void *>(a->output), static_cast<const void *>(a->activation_scratch),
         static_cast<const void *>(a->arrival_counter), static_cast<const void *>(a->sticky_error)}) {
        if (reinterpret_cast<uintptr_t>(p) % alignof(uint32_t)) return 0;
    }
    uint64_t gate_bytes, up_bytes, down_bytes;
    const uint64_t down_block = a->down_type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K);
    if (!matrix_span(a->gate_row_stride, a->gate_expert_stride, 16 * sizeof(block_q4_K), intermediate, 4, gate_bytes) ||
        !matrix_span(a->up_row_stride, a->up_expert_stride, 16 * sizeof(block_q4_K), intermediate, 4, up_bytes) ||
        !matrix_span(a->down_row_stride, a->down_expert_stride, 2 * down_block, hidden, a->down_type == GGML_TYPE_Q4_K ? 4 : 2, down_bytes) ||
        a->routing_stride < sizeof(float) || a->routing_stride % sizeof(float) ||
        a->routing_stride > uint64_t(PTRDIFF_MAX) / selected) return 0;
    const memory_span reads[] = {{a->gate, gate_bytes}, {a->up, up_bytes}, {a->down, down_bytes},
        {a->input, hidden * sizeof(float)}, {a->selected_experts, selected * sizeof(int32_t)},
        {a->routing_weights, (selected - 1) * a->routing_stride + sizeof(float)}};
    const memory_span writes[] = {{a->output, hidden * sizeof(float)},
        {a->activation_scratch, selected * intermediate * sizeof(float)}, {a->arrival_counter, 4}, {a->sticky_error, 4}};
    for (const auto & r : reads) {
        if (reinterpret_cast<uintptr_t>(r.pointer) > UINTPTR_MAX - r.bytes) return 0;
    }
    for (const auto & w : writes) {
        if (reinterpret_cast<uintptr_t>(w.pointer) > UINTPTR_MAX - w.bytes) return 0;
        for (const auto & r : reads) if (overlaps(w, r)) return 0;
        for (const auto & other : writes) if (w.pointer != other.pointer && overlaps(w, other)) return 0;
    }
    // Identical writable pointers also overlap and are never permitted.
    for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) if (writes[i].pointer == writes[j].pointer) return 0;
    return 1;
}

extern "C" hipError_t qwen3moe_full_moe_query(const qwen3moe_full_moe_args * a, qwen3moe_full_moe_launch_info * info) {
    if (!info || !qwen3moe_full_moe_supported(a)) return hipErrorInvalidValue;
    *info = {};
    hipError_t error = hipGetDevice(&info->device);
    if (error != hipSuccess) return error;
    hipDeviceProp_t props{};
    error = hipGetDeviceProperties(&props, info->device);
    if (error != hipSuccess) return error;
    if (props.warpSize != 32 || std::strncmp(props.gcnArchName, "gfx1151", 7) != 0 || !props.cooperativeLaunch) return hipErrorNotSupported;
    int wall_clock_khz = 0;
    error = hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, info->device);
    if (error != hipSuccess) return error;
    if (wall_clock_khz <= 0) return hipErrorNotSupported;
    info->multiprocessors = props.multiProcessorCount;
    info->grid_blocks = a->grid_blocks;
    info->down_type = a->down_type;
    info->gate_rows_per_wave = a->gate_rows_per_wave;
    info->down_rows = down_rows;
    info->down_mapping = QWEN3MOE_FULL_MOE_DOWN_MAPPING;
    info->barrier_timeout_ticks = uint64_t(wall_clock_khz) * 10; // 10 ms; hard poll cap also applies
    if (a->down_type == GGML_TYPE_Q6_K) {
        error = a->gate_rows_per_wave == 2 ? occupancy<true, 2>(&info->active_blocks_per_multiprocessor)
                                           : occupancy<true, 4>(&info->active_blocks_per_multiprocessor);
    } else {
        error = a->gate_rows_per_wave == 2 ? occupancy<false, 2>(&info->active_blocks_per_multiprocessor)
                                           : occupancy<false, 4>(&info->active_blocks_per_multiprocessor);
    }
    if (error != hipSuccess) return error;
    if (info->active_blocks_per_multiprocessor <= 0 ||
        int64_t(info->active_blocks_per_multiprocessor) * info->multiprocessors < a->grid_blocks) return hipErrorCooperativeLaunchTooLarge;
    return hipSuccess;
}

extern "C" hipError_t qwen3moe_full_moe_enqueue(const qwen3moe_full_moe_args * a,
                                               const qwen3moe_full_moe_launch_info * info, hipStream_t stream) {
    if (!info || !qwen3moe_full_moe_supported(a) || info->device < 0 ||
        info->grid_blocks != a->grid_blocks || info->down_type != a->down_type ||
        info->gate_rows_per_wave != a->gate_rows_per_wave || info->down_rows != down_rows ||
        info->down_mapping != QWEN3MOE_FULL_MOE_DOWN_MAPPING || !info->barrier_timeout_ticks ||
        info->active_blocks_per_multiprocessor <= 0 || info->multiprocessors <= 0 ||
        int64_t(info->active_blocks_per_multiprocessor) * info->multiprocessors < a->grid_blocks) {
        return hipErrorInvalidValue;
    }
    hipError_t error = hipMemsetAsync(a->arrival_counter, 0, sizeof(uint32_t), stream);
    if (error != hipSuccess) return error;
    if (a->down_type == GGML_TYPE_Q6_K) {
        return a->gate_rows_per_wave == 2 ? submit<true, 2>(*a, info->barrier_timeout_ticks, stream)
                                         : submit<true, 4>(*a, info->barrier_timeout_ticks, stream);
    }
    return a->gate_rows_per_wave == 2 ? submit<false, 2>(*a, info->barrier_timeout_ticks, stream)
                                     : submit<false, 4>(*a, info->barrier_timeout_ticks, stream);
}

extern "C" hipError_t qwen3moe_full_moe_launch(const qwen3moe_full_moe_args * a, hipStream_t stream) {
    qwen3moe_full_moe_launch_info info{};
    const hipError_t error = qwen3moe_full_moe_query(a, &info);
    return error == hipSuccess ? qwen3moe_full_moe_enqueue(a, &info, stream) : error;
}
