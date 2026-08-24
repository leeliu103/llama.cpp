import triton
import triton.language as tl


@triton.jit
def gptoss_q8_0_w8a16_qkv_bias(
    out_ptr,
    a_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    m_size,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    QK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m_size, BLOCK_M)
    num_pid_n = tl.cdiv(N_SIZE, BLOCK_N)
    num_pid_in_group = GROUP_M_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M_SIZE
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K_SIZE, BLOCK_K):
        k = k_base + offs_k
        k_mask = k < K_SIZE
        a = tl.load(
            a_ptr + offs_m[:, None] * K_SIZE + k[None, :],
            mask=(offs_m[:, None] < m_size) & k_mask[None, :],
            other=0.0,
        )
        w_q = tl.load(
            w_ptr + offs_n[:, None] * K_SIZE + k[None, :],
            mask=(offs_n[:, None] < N_SIZE) & k_mask[None, :],
            other=0,
        )
        scale_offsets = k_base // QK_SIZE + tl.arange(
            0, BLOCK_K // QK_SIZE
        )
        scale_blocks = tl.load(
            scale_ptr
            + offs_n[:, None] * (K_SIZE // QK_SIZE)
            + scale_offsets[None, :],
            mask=(offs_n[:, None] < N_SIZE)
            & (scale_offsets[None, :] < K_SIZE // QK_SIZE),
            other=0.0,
        )
        scale = tl.reshape(
            tl.broadcast_to(
                scale_blocks[:, :, None],
                (BLOCK_N, BLOCK_K // QK_SIZE, QK_SIZE),
            ),
            (BLOCK_N, BLOCK_K),
        )
        w = (w_q.to(tl.float16) * scale).trans(1, 0)
        acc = tl.dot(a, w, acc=acc)

    out_mask = (offs_m[:, None] < m_size) & (
        offs_n[None, :] < N_SIZE
    )
    bias = tl.load(
        bias_ptr + offs_n,
        mask=offs_n < N_SIZE,
        other=0.0,
    )
    tl.store(
        out_ptr + offs_m[:, None] * N_SIZE + offs_n[None, :],
        acc + bias[None, :],
        mask=out_mask,
    )


@triton.jit
def gptoss_q8_0_w8a16_attn_output_bias_residual(
    out_ptr,
    a_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    residual_ptr,
    m_size,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    QK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m_size, BLOCK_M)
    num_pid_n = tl.cdiv(N_SIZE, BLOCK_N)
    num_pid_in_group = GROUP_M_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M_SIZE
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K_SIZE, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * K_SIZE + k_base + offs_k[None, :],
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        w_q = tl.load(
            w_ptr + offs_n[:, None] * K_SIZE + k_base + offs_k[None, :],
            mask=offs_n[:, None] < N_SIZE,
            other=0,
        )
        scale_blocks = tl.load(
            scale_ptr
            + offs_n[:, None] * (K_SIZE // QK_SIZE)
            + k_base // QK_SIZE
            + tl.arange(0, BLOCK_K // QK_SIZE)[None, :],
            mask=offs_n[:, None] < N_SIZE,
            other=0.0,
        )
        scale = tl.reshape(
            tl.broadcast_to(
                scale_blocks[:, :, None],
                (BLOCK_N, BLOCK_K // QK_SIZE, QK_SIZE),
            ),
            (BLOCK_N, BLOCK_K),
        )
        w = (w_q.to(tl.float16) * scale).trans(1, 0)
        acc = tl.dot(a, w, acc=acc)

    out_mask = (offs_m[:, None] < m_size) & (
        offs_n[None, :] < N_SIZE
    )
    bias = tl.load(
        bias_ptr + offs_n,
        mask=offs_n < N_SIZE,
        other=0.0,
    )
    residual = tl.load(
        residual_ptr + offs_m[:, None] * N_SIZE + offs_n[None, :],
        mask=out_mask,
        other=0.0,
    )
    tl.store(
        out_ptr + offs_m[:, None] * N_SIZE + offs_n[None, :],
        acc + bias[None, :] + residual,
        mask=out_mask,
    )
