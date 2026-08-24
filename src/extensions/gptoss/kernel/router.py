import triton
import triton.language as tl


@triton.jit
def gptoss_router(
    output_ptr,
    activation_ptr,
    weight_ptr,
    m_size,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K_SIZE, BLOCK_K):
        activation = tl.load(
            activation_ptr
            + offs_m[:, None] * K_SIZE
            + k_base
            + offs_k[None, :],
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + offs_n[:, None] * K_SIZE
            + k_base
            + offs_k[None, :],
        )
        acc = tl.dot(
            activation,
            weight.trans(1, 0),
            acc=acc,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    tl.store(
        output_ptr + offs_m[:, None] * N_SIZE + offs_n[None, :],
        acc,
        mask=offs_m[:, None] < m_size,
    )
