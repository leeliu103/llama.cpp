#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>

constexpr uint32_t gptoss_fa_tile_size = 32;

hipError_t gptoss_q8_qkv_launch(hipFunction_t  function,
                                __half *        output,
                                const __half *  activation,
                                const int8_t *  values,
                                const __half *  scales,
                                const float *   bias,
                                uint32_t        n_tokens,
                                hipStream_t     stream);

hipError_t gptoss_q8_attention_output_launch(hipFunction_t  function,
                                             float *        output,
                                             const __half * activation,
                                             const int8_t * values,
                                             const __half * scales,
                                             const float *  bias,
                                             const float *  residual,
                                             uint32_t       n_tokens,
                                             hipStream_t    stream);

hipError_t gptoss_router_launch(hipFunction_t  function,
                                float *        output,
                                const float *  activation,
                                const float *  weight,
                                uint32_t       n_tokens,
                                hipStream_t    stream);

hipError_t gptoss_fa_launch(hipFunction_t  function,
                            __half *        output,
                            const __half *  query,
                            const __half *  key_cache,
                            const __half *  value_cache,
                            const float *   sinks,
                            const int32_t * block_table,
                            const int32_t * seq_lens,
                            int64_t         block_table_stride,
                            const int32_t * cu_seqlens_q,
                            uint32_t        n_sequences,
                            uint32_t        n_tokens,
                            hipStream_t     stream);

hipError_t gptoss_ogs_w13_launch(hipFunction_t  function,
                                 __half *        output,
                                 const __half *  activation,
                                 const uint8_t * values,
                                 const uint8_t * scales,
                                 const float *   bias,
                                 const int32_t * gather_indices,
                                 const int32_t * expert_counts,
                                 const int32_t * route_offsets,
                                 const int32_t * block_offsets,
                                 const int32_t * block_schedule,
                                 uint32_t        schedule_capacity,
                                 hipStream_t     stream);

hipError_t gptoss_ogs_w2_launch(hipFunction_t  function,
                                float *         output,
                                const __half *  activation,
                                const uint8_t * values,
                                const uint8_t * scales,
                                const float *   bias,
                                const int32_t * scatter_indices,
                                uint32_t        route_count,
                                const int32_t * expert_counts,
                                const int32_t * route_offsets,
                                const int32_t * block_offsets,
                                const int32_t * block_schedule,
                                uint32_t        schedule_capacity,
                                hipStream_t     stream);
