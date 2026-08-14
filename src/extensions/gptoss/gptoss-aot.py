#!/usr/bin/env python3

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["AITER_AOT_IMPORT"] = "1"

import triton
from aiter.ops.triton._triton_kernels.attention.mha import _attn_fwd
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
from triton_kernels.matmul_details._matmul import _matmul as _ogs_matmul
from triton_kernels.specialize import ClosureArg, FnSpecs, SpecializationModule
from triton_kernels.swiglu import swiglu_fn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kernel"))

from matmul_q8_0_w8a16 import (
    gptoss_q8_0_w8a16_attn_output_bias_residual,
    gptoss_q8_0_w8a16_qkv_bias,
)


@dataclass(frozen=True)
class KernelSpec:
    output_name: str
    kernel: object
    runtime_types: dict[str, str]
    kernel_constants: dict[str, object]
    compiler_options: dict[str, object]
    assume_32_bit_pointer_range: bool


def _make_ogs_kernels():
    ogs = SpecializationModule(
        "gptoss_ogs",
        [("_matmul", _ogs_matmul)],
        {
            "activation": ClosureArg(
                "ACTIVATION_FN",
                "activation_fn_args",
            ),
            "epilogue": ClosureArg(
                "EPILOGUE_FN",
                "epilogue_fn_args",
            ),
        },
    )
    no_fusion = FnSpecs.default()
    swiglu = FnSpecs(
        "swiglu",
        swiglu_fn,
        ("alpha", "limit"),
        reduction_n=2,
    )
    w13_kernel = ogs.get(
        activation=swiglu,
        epilogue=no_fusion,
    )._matmul
    w2_kernel = ogs.get(
        activation=no_fusion,
        epilogue=no_fusion,
    )._matmul
    return w13_kernel, w2_kernel


def _matmul_compiler_options(num_warps, num_stages, waves_per_eu):
    return {
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
        "enable_fp_fusion": True,
    }


def _fa_compiler_options(num_warps, num_stages, waves_per_eu):
    return {
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "num_ctas": 1,
        "matrix_instr_nonkdim": 0,
        "kpack": 1,
    }


def _fa_kernel_constants(
    *,
    query_heads,
    key_value_heads,
    head_size,
    block_m,
    block_n,
    sliding_window,
):
    query_size = query_heads * head_size
    key_value_size = key_value_heads * head_size

    return {
        "descale_q_ptr": None,
        "descale_k_ptr": None,
        "descale_v_ptr": None,
        "alibi_slopes_ptr": None,
        "s_dmask_ptr": None,
        "dropout_mask_ptr": None,
        "softmax_lse_ptr": None,
        "stride_qz_in": 0,
        "stride_qh_in": head_size,
        "stride_qm_in": query_size,
        "stride_qk_in": 1,
        "stride_kz_in": 0,
        "stride_kh_in": head_size,
        "stride_kn_in": key_value_size,
        "stride_kk_in": 1,
        "stride_vz_in": 0,
        "stride_vh_in": head_size,
        "stride_vn_in": key_value_size,
        "stride_vk_in": 1,
        "stride_descale_q_z_in": 0,
        "stride_descale_k_z_in": 0,
        "stride_descale_v_z_in": 0,
        "stride_oz_in": 0,
        "stride_oh_in": head_size,
        "stride_om_in": query_size,
        "stride_on_in": 1,
        "stride_alibi_z_in": 0,
        "stride_alibi_h_in": 0,
        "stride_sd_z_in": 0,
        "stride_sd_h_in": 0,
        "stride_sd_m_in": 0,
        "stride_sd_n_in": 0,
        "stride_lse_z_in": 0,
        "stride_lse_h_in": 0,
        "stride_lse_m_in": 1,
        "dropout_p": 0.0,
        "philox_seed": 0,
        "philox_offset_base_in": 0,
        "IS_CAUSAL": True,
        "NUM_Q_HEADS": query_heads,
        "NUM_K_HEADS": key_value_heads,
        "PRELOAD_V": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_DMODEL": head_size,
        "BLOCK_DMODEL_POW2": head_size,
        "BLOCK_DMODEL_PE": 0,
        "RETURN_SCORES": False,
        "ENABLE_DROPOUT": False,
        "IS_FP8": False,
        "FP8_MAX": 65504.0,
        "VARLEN": True,
        "NUM_XCD": 1,
        "SWIZZLE": "default",
        "USE_INT64_STRIDES": False,
        "ENABLE_SINK": True,
        "SLIDING_WINDOW": sliding_window,
        "HEAD_STRIDE_ALIGNED_8": True,
    }


def _ogs_kernel_constants(
    *,
    n_size,
    k_size,
    physical_n_size,
    padded_k_size,
    mxfp4_block_size,
    expert_count,
    activation_reduction,
    block_m,
    block_n,
    block_k,
    group_m,
    xcd_swizzle,
):
    output_size = n_size // activation_reduction
    packed_k_size = padded_k_size // 2
    scale_k_size = padded_k_size // mxfp4_block_size

    return {
        "stride_y_k": output_size,
        "stride_y_z": 0,
        "stride_y_m": output_size,
        "stride_y_n": 1,
        "YExpectedScale": None,
        "YActualScale": None,
        "YChecksumScale": None,
        "stride_y_mx_k": None,
        "stride_y_mx_z": None,
        "stride_y_mx_m": None,
        "stride_y_mx_n": None,
        "stride_x_z": 0,
        "stride_x_m": k_size,
        "stride_x_k": 1,
        "X_TRANSPOSE": False,
        "XScale": None,
        "XMxScale": None,
        "stride_x_mx_z": None,
        "stride_x_mx_m": None,
        "stride_x_mx_k": None,
        "stride_w_e": physical_n_size * packed_k_size,
        "stride_w_k": 1,
        "stride_w_n": packed_k_size,
        "W_TRANSPOSE": True,
        "WScale": None,
        "stride_w_mx_e": physical_n_size * scale_k_size,
        "stride_w_mx_k": 1,
        "stride_w_mx_n": scale_k_size,
        "OutAcc": None,
        "stride_acc_z": None,
        "stride_acc_m": None,
        "stride_acc_n": None,
        "OutAccScale": None,
        "Y_ACC_IS_Y": None,
        "stride_b_e": n_size,
        "M": None,
        "N": n_size,
        "K": k_size,
        "K_W": k_size,
        "Betas": None,
        "Gammas": None,
        "RAGGED_DIMENSION": "M",
        "X_EXPECTED_SLICE_SIZE": None,
        "X_SLICE_SIZES_DIVISIBILITY": 1,
        "WSliceSizes": None,
        "WSliceOffs": None,
        "WBlockOffs": None,
        "WBlockSchedule": None,
        "W_EXPECTED_SLICE_SIZE": None,
        "_W_SLICE_SIZES_DIVISIBILITY": None,
        "batch_size": 1,
        "grid_n": (n_size + block_n - 1) // block_n,
        "out_alpha": None,
        "ACTIVATION_REDUCTION_N": activation_reduction,
        "N_EXPTS_TOT": expert_count,
        "MAX_NUM_IMPRECISE_ACC": None,
        "ALLOW_TF32": True,
        "FLEXPOINT_SATURATE_INF": False,
        "PER_BATCH_W_SCALE": False,
        "PER_BATCH_OUT_SCALE": False,
        "PER_BATCH_ACC_SCALE": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "XCD_SWIZZLE": xcd_swizzle,
        "SWIZZLE_MX_VALUE": "STRIDED",
        "SWIZZLE_MX_SCALE": "STRIDED",
        "EPILOGUE_SUBTILE": 1,
        "EVEN_K": k_size % block_k == 0,
        "SPLIT_K": 1,
        "W_CACHE_MODIFIER": None,
        "NUM_SMS": 0,
        "X_TMA_MODE": None,
        "Y_TMA_MODE": None,
        "TOKENS_PER_EXPT_FOR_ANNOTATION": None,
        "UPCAST_INDICES": False,
        "SWAP_XW": False,
        "IS_EPILOGUE_QUANT_MXFP8": False,
        "FLATTEN_LOOPS": True,
        "pYPtrs": None,
        "map_dst_coord": None,
        "all_writes_issued": None,
        "reduce_rank": 0,
        "n_reduce_shards": 1,
    }


def _build_kernel_specs():
    ogs_w13, ogs_w2 = _make_ogs_kernels()

    return (
        KernelSpec(
            output_name="q8_qkv.hsaco",
            kernel=gptoss_q8_0_w8a16_qkv_bias,
            runtime_types={
                "out_ptr": "*fp16",
                "a_ptr": "*fp16",
                "w_ptr": "*i8",
                "scale_ptr": "*fp16",
                "bias_ptr": "*fp32",
                "m_size": "i32",
            },
            kernel_constants={
                "N_SIZE": 5120,
                "K_SIZE": 2880,
                "QK_SIZE": 32,
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "GROUP_M_SIZE": 4,
            },
            compiler_options=_matmul_compiler_options(
                num_warps=4,
                num_stages=1,
                waves_per_eu=0,
            ),
            assume_32_bit_pointer_range=True,
        ),
        KernelSpec(
            output_name="q8_attn_out.hsaco",
            kernel=gptoss_q8_0_w8a16_attn_output_bias_residual,
            runtime_types={
                "out_ptr": "*fp32",
                "a_ptr": "*fp16",
                "w_ptr": "*i8",
                "scale_ptr": "*fp16",
                "bias_ptr": "*fp32",
                "residual_ptr": "*fp32",
                "m_size": "i32",
            },
            kernel_constants={
                "N_SIZE": 2880,
                "K_SIZE": 4096,
                "QK_SIZE": 32,
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "GROUP_M_SIZE": 4,
            },
            compiler_options=_matmul_compiler_options(
                num_warps=4,
                num_stages=1,
                waves_per_eu=0,
            ),
            assume_32_bit_pointer_range=True,
        ),
        KernelSpec(
            output_name="fa_full.hsaco",
            kernel=_attn_fwd,
            runtime_types={
                "q_ptr": "*fp16",
                "k_ptr": "*fp16",
                "v_ptr": "*fp16",
                "out_ptr": "*fp16",
                "sink_ptr": "*fp32",
                "sm_scale": "fp32",
                "cu_seqlens_q": "*i32",
                "cu_seqlens_k": "*i32",
                "SEQLEN_Q": "i32",
                "SEQLEN_K": "i32",
                "BATCH": "i32",
            },
            kernel_constants=_fa_kernel_constants(
                query_heads=64,
                key_value_heads=8,
                head_size=64,
                block_m=64,
                block_n=32,
                sliding_window=0,
            ),
            compiler_options=_fa_compiler_options(
                num_warps=2,
                num_stages=1,
                waves_per_eu=1,
            ),
            assume_32_bit_pointer_range=False,
        ),
        KernelSpec(
            output_name="fa_sw128.hsaco",
            kernel=_attn_fwd,
            runtime_types={
                "q_ptr": "*fp16",
                "k_ptr": "*fp16",
                "v_ptr": "*fp16",
                "out_ptr": "*fp16",
                "sink_ptr": "*fp32",
                "sm_scale": "fp32",
                "cu_seqlens_q": "*i32",
                "cu_seqlens_k": "*i32",
                "SEQLEN_Q": "i32",
                "SEQLEN_K": "i32",
                "BATCH": "i32",
            },
            kernel_constants=_fa_kernel_constants(
                query_heads=64,
                key_value_heads=8,
                head_size=64,
                block_m=32,
                block_n=32,
                sliding_window=128 - 1,
            ),
            compiler_options=_fa_compiler_options(
                num_warps=2,
                num_stages=1,
                waves_per_eu=0,
            ),
            assume_32_bit_pointer_range=False,
        ),
        KernelSpec(
            output_name="ogs_w13.hsaco",
            kernel=ogs_w13,
            runtime_types={
                "Y": "*fp16",
                "YPtr": "*fp16",
                "X": "*fp16",
                "XPtr": "*fp16",
                "W": "*u8",
                "WPtr": "*u8",
                "WMxScale": "*u8",
                "B": "*fp32",
                "GatherIndx": "*i32",
                "XSliceSizes": "*i32",
                "XSliceOffs": "*i32",
                "XBlockOffs": "*i32",
                "XBlockSchedule": "*i32",
                "grid_m": "i32",
            },
            kernel_constants={
                **_ogs_kernel_constants(
                    n_size=5760,
                    k_size=2880,
                    physical_n_size=6144,
                    padded_k_size=3072,
                    mxfp4_block_size=32,
                    expert_count=32,
                    activation_reduction=2,
                    block_m=64,
                    block_n=128,
                    block_k=128,
                    group_m=4,
                    xcd_swizzle=8,
                ),
                "WriteBackIndx": None,
                "writeback_size": None,
                "alpha": 1.702,
                "limit": 7.0,
            },
            compiler_options=_matmul_compiler_options(
                num_warps=4,
                num_stages=1,
                waves_per_eu=0,
            ),
            assume_32_bit_pointer_range=True,
        ),
        KernelSpec(
            output_name="ogs_w2.hsaco",
            kernel=ogs_w2,
            runtime_types={
                "Y": "*fp32",
                "YPtr": "*fp32",
                "X": "*fp16",
                "XPtr": "*fp16",
                "W": "*u8",
                "WPtr": "*u8",
                "WMxScale": "*u8",
                "B": "*fp32",
                "WriteBackIndx": "*i32",
                "writeback_size": "i32",
                "XSliceSizes": "*i32",
                "XSliceOffs": "*i32",
                "XBlockOffs": "*i32",
                "XBlockSchedule": "*i32",
                "grid_m": "i32",
            },
            kernel_constants={
                **_ogs_kernel_constants(
                    n_size=2880,
                    k_size=2880,
                    physical_n_size=3072,
                    padded_k_size=3072,
                    mxfp4_block_size=32,
                    expert_count=32,
                    activation_reduction=1,
                    block_m=64,
                    block_n=128,
                    block_k=128,
                    group_m=4,
                    xcd_swizzle=8,
                ),
                "GatherIndx": None,
            },
            compiler_options=_matmul_compiler_options(
                num_warps=4,
                num_stages=1,
                waves_per_eu=0,
            ),
            assume_32_bit_pointer_range=True,
        ),
    )


def _compile_kernel(output_dir, backend, target, spec):
    runtime_names = set(spec.runtime_types)
    constant_names = set(spec.kernel_constants)
    overlap = runtime_names & constant_names
    if overlap:
        raise RuntimeError(
            f"{spec.output_name} arguments are both runtime and constant: "
            f"{sorted(overlap)}"
        )

    names = set(spec.kernel.arg_names)
    configured = runtime_names | constant_names
    if names != configured:
        raise RuntimeError(
            f"unexpected {spec.output_name} kernel signature: "
            f"missing={sorted(names - configured)}, "
            f"extra={sorted(configured - names)}"
        )

    signature = {
        name: (
            spec.runtime_types[name]
            if name in spec.runtime_types
            else "constexpr"
        )
        for name in spec.kernel.arg_names
    }
    attrs = {}
    for index, name in enumerate(spec.kernel.arg_names):
        if spec.runtime_types.get(name, "").startswith("*"):
            attrs[(index,)] = [["tt.divisibility", 16]]
            if spec.assume_32_bit_pointer_range:
                attrs[(index,)].append(["tt.pointer_range", 32])

    source = ASTSource(
        fn=spec.kernel,
        signature=signature,
        constexprs=spec.kernel_constants,
        attrs=attrs,
    )
    options = backend.parse_options(spec.compiler_options)
    compiled = triton.compile(
        source,
        target=target,
        options=options.__dict__,
    )
    if (
        getattr(compiled.metadata, "global_scratch_size", 0)
        or compiled.metadata.profile_scratch_size
    ):
        raise RuntimeError(f"{spec.output_name}: unexpected Triton scratch")

    (output_dir / spec.output_name).write_bytes(
        compiled.asm[backend.binary_ext]
    )


def _compile_arch(arch, specs):
    output_dir = ROOT / "build" / arch
    output_dir.mkdir(parents=True, exist_ok=True)

    target = GPUTarget("hip", arch, 32)
    backend = make_backend(target)
    for spec in specs:
        _compile_kernel(output_dir, backend, target, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        action="append",
        choices=("gfx1100", "gfx1201"),
        required=True,
    )
    args = parser.parse_args()

    specs = _build_kernel_specs()
    for arch in dict.fromkeys(args.arch):
        _compile_arch(arch, specs)


if __name__ == "__main__":
    main()
