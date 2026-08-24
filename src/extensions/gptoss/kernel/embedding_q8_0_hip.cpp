#include "../gptoss-config.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t hidden_size      = gptoss_hidden_size;
constexpr uint32_t quant_block_size = gptoss_quant_block_size;

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[quant_block_size];
};

static_assert(sizeof(block_q8_0) == 34);

}  // namespace

__global__ void gptoss_embedding_q8_0_kernel(float * output, const uint8_t * weight, const int32_t * tokens) {
    const uint32_t token = blockIdx.x;
    const auto *   row   = reinterpret_cast<const block_q8_0 *>(weight) +
                       static_cast<uint64_t>(tokens[token]) * (hidden_size / quant_block_size);

    for (uint32_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const block_q8_0 & block = row[column / quant_block_size];
        const float        scale = __half2float(__ushort_as_half(block.d));
        output[static_cast<uint64_t>(token) * hidden_size + column] =
            scale * static_cast<float>(block.qs[column % quant_block_size]);
    }
}
