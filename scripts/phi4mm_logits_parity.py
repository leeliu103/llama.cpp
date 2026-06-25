#!/usr/bin/env python3
"""Compare Phi-4-MM llama.cpp logits against the official HF model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phi4mm_hf_baseline import (
    DEFAULT_MODEL_PATH,
    configure_determinism,
    load_image,
    load_processor_and_model,
    move_tensors,
    prepare_inputs,
    select_device,
)


DEFAULT_PREPROC_BASELINE_DIR = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct-hf-baseline")
DEFAULT_TEXT_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/phi4mm-text-bf16.gguf")
DEFAULT_MMPROJ_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/mmproj-phi4mm-f32.gguf")
DEFAULT_HF_DIR = Path("/tmp/phi4mm_hf_logits")
DEFAULT_CPP_DIR = Path("/tmp/phi4mm_cpp_logits")
DEFAULT_USER_PROMPT = "What is shown in this image?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preproc-baseline-dir", type=Path, default=DEFAULT_PREPROC_BASELINE_DIR)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--model", type=Path, default=DEFAULT_TEXT_GGUF)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ_GGUF)
    parser.add_argument("--hf-output-dir", type=Path, default=DEFAULT_HF_DIR)
    parser.add_argument("--cpp-output-dir", type=Path, default=DEFAULT_CPP_DIR)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--forced-tokens", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--attn-implementation", default="eager", choices=("auto", "eager", "sdpa", "flash_attention_2"))
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=4096,
        help=(
            "llama.cpp context size for the text decoder. Phi-4-MM HF uses LongRoPE short factors until "
            "the actual sequence length exceeds original_max_position_embeddings=4096; defaulting the "
            "parity run to 4096 prevents llama.cpp from selecting long factors for short prompts."
        ),
    )
    parser.add_argument(
        "--hf-decode-mode",
        choices=("cache", "full"),
        default="cache",
        help=(
            "HF reference decode mode. 'cache' matches llama.cpp incremental KV-cache decode; "
            "'full' recomputes the whole prefix each step and is useful only for diagnostics. Default: cache."
        ),
    )
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--tie-margin-threshold",
        type=float,
        default=0.25,
        help=(
            "For argmax mismatches, mark as near_tie when either model's winner is within this "
            "logit margin of the other model's winner. Default: 0.25."
        ),
    )
    parser.add_argument("--skip-hf-run", action="store_true")
    parser.add_argument("--skip-cpp-run", action="store_true")
    return parser.parse_args()


def load_samples(preproc_baseline_dir: Path, requested: list[str]) -> list[dict[str, Any]]:
    manifest = json.loads((preproc_baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if requested:
        wanted = set(requested)
        samples = [sample for sample in samples if sample["label"] in wanted]
        missing = sorted(wanted - {sample["label"] for sample in samples})
        if missing:
            raise ValueError(f"Requested sample(s) not found: {', '.join(missing)}")
    return samples


def hf_prompt(user_prompt: str) -> str:
    return f"<|user|><|image_1|>{user_prompt}<|end|><|assistant|>"


def parse_forced_tokens(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run_hf_logits(args: argparse.Namespace, samples: list[dict[str, Any]]) -> None:
    # Reuse the baseline loader's argparse-shaped object.
    args.preprocess_only = False
    args.num_logits_to_keep = 1
    configure_determinism(args.seed)
    device = select_device(args.device)
    processor, model = load_processor_and_model(args, device)
    assert model is not None

    args.hf_output_dir.mkdir(parents=True, exist_ok=True)
    forced_from_cli = parse_forced_tokens(args.forced_tokens)
    prompt = hf_prompt(args.user_prompt)

    for sample in samples:
        label = sample["label"]
        sample_dir = args.hf_output_dir / label
        sample_dir.mkdir(parents=True, exist_ok=True)

        image = load_image(Path(sample["image_path"]))
        model_inputs_cpu, _image_inputs = prepare_inputs(processor, prompt, image)
        image.close()

        model_inputs = move_tensors(model_inputs_cpu, device)
        forced_tokens: list[int] = list(forced_from_cli)
        logits_list: list[torch.Tensor] = []

        with torch.inference_mode():
            generated: list[int] = []
            n_steps = len(forced_tokens) if forced_tokens else args.max_new_tokens

            if args.hf_decode_mode == "cache":
                outputs = model(
                    **model_inputs,
                    use_cache=True,
                    return_dict=True,
                    num_logits_to_keep=1,
                )
                logits = outputs.logits[:, -1, :].squeeze(0).detach().float().cpu()
                logits_list.append(logits)
                past_key_values = outputs.past_key_values
                attention_mask = model_inputs["attention_mask"]

                for step in range(n_steps):
                    if forced_from_cli:
                        token = forced_tokens[step]
                    else:
                        token = int(torch.argmax(logits).item())
                        forced_tokens.append(token)
                    generated.append(token)

                    input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
                    token_mask = torch.ones_like(input_ids, dtype=attention_mask.dtype, device=device)
                    attention_mask = torch.cat([attention_mask, token_mask], dim=1)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        input_mode=model_inputs["input_mode"],
                        use_cache=True,
                        return_dict=True,
                        num_logits_to_keep=1,
                    )
                    logits = outputs.logits[:, -1, :].squeeze(0).detach().float().cpu()
                    logits_list.append(logits)
                    past_key_values = outputs.past_key_values

            else:
                for step in range(n_steps + 1):
                    if generated:
                        gen = torch.tensor([generated], dtype=torch.long, device=device)
                        input_ids = torch.cat([model_inputs["input_ids"], gen], dim=1)
                        gen_mask = torch.ones_like(gen, dtype=model_inputs["attention_mask"].dtype, device=device)
                        attention_mask = torch.cat([model_inputs["attention_mask"], gen_mask], dim=1)
                    else:
                        input_ids = model_inputs["input_ids"]
                        attention_mask = model_inputs["attention_mask"]

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_image_embeds=model_inputs["input_image_embeds"],
                        image_sizes=model_inputs["image_sizes"],
                        image_attention_mask=model_inputs["image_attention_mask"],
                        input_mode=model_inputs["input_mode"],
                        use_cache=False,
                        return_dict=True,
                        num_logits_to_keep=1,
                    )
                    logits = outputs.logits[:, -1, :].squeeze(0).detach().float().cpu()
                    logits_list.append(logits)

                    if step == n_steps:
                        break
                    if forced_from_cli:
                        token = forced_tokens[step]
                    else:
                        token = int(torch.argmax(logits).item())
                        forced_tokens.append(token)
                    generated.append(token)

        for i, logits in enumerate(logits_list):
            logits.numpy().astype(np.float32).tofile(sample_dir / f"logits_{i:03d}.f32")

        (sample_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "prompt": prompt,
                    "user_prompt": args.user_prompt,
                    "hf_decode_mode": args.hf_decode_mode,
                    "input_ids_shape": list(model_inputs_cpu["input_ids"].shape),
                    "forced_tokens": forced_tokens,
                    "files": [
                        {
                            "step": i,
                            "file": f"logits_{i:03d}.f32",
                            "dtype": "float32",
                            "shape": list(logits.shape),
                        }
                        for i, logits in enumerate(logits_list)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def run_cpp_logits(args: argparse.Namespace, sample: dict[str, Any], forced_tokens: list[int]) -> None:
    sample_dir = args.cpp_output_dir / sample["label"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    cli_bin = args.build_dir / "bin" / "llama-mtmd-cli"
    cmd = [
        str(cli_bin),
        "-m", str(args.model),
        "--mmproj", str(args.mmproj),
        "--no-warmup",
        "--no-mmproj-offload",
        "--flash-attn", "off",
        "-ngl", "0",
        "-c", str(args.ctx_size),
        "-p", args.user_prompt,
        "--image", str(sample["image_path"]),
        "-n", "0",
        "--temp", "0",
        "--top-k", "1",
        "--seed", str(args.seed),
        "--jinja",
    ]
    env = os.environ.copy()
    env["MTMD_DEBUG_PHI4MM_LOGITS_DUMP"] = str(sample_dir)
    env["MTMD_DEBUG_PHI4MM_FORCE_TOKENS"] = ",".join(str(token) for token in forced_tokens)
    subprocess.run(cmd, check=True, env=env)


def load_hf_sample(hf_dir: Path, label: str) -> tuple[dict[str, Any], list[np.ndarray]]:
    sample_dir = hf_dir / label
    manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
    logits = [
        np.fromfile(sample_dir / item["file"], dtype=np.float32).reshape(item["shape"])
        for item in manifest["files"]
    ]
    return manifest, logits


def load_cpp_sample(cpp_dir: Path, label: str) -> tuple[dict[str, Any], list[np.ndarray]]:
    sample_dir = cpp_dir / label
    manifest = json.loads((sample_dir / "logits_manifest.json").read_text(encoding="utf-8"))
    logits = [
        np.fromfile(sample_dir / item["file"], dtype=np.float32).reshape(item["shape"])
        for item in manifest["files"]
    ]
    return manifest, logits


def logits_kl(hf: np.ndarray, cpp: np.ndarray) -> float:
    hf_t = torch.from_numpy(hf.astype(np.float64))
    cpp_t = torch.from_numpy(cpp.astype(np.float64))
    log_p = torch.log_softmax(hf_t, dim=-1)
    log_q = torch.log_softmax(cpp_t, dim=-1)
    p = torch.exp(log_p)
    return float(torch.sum(p * (log_p - log_q)).item())


def top2_info(logits: np.ndarray) -> tuple[int, float, int, float, float]:
    if logits.ndim != 1 or logits.size < 2:
        raise ValueError(f"expected 1D logits with at least 2 values, got shape={logits.shape}")
    top2 = np.argpartition(logits, -2)[-2:]
    top2 = top2[np.argsort(logits[top2])[::-1]]
    top1 = int(top2[0])
    top2_id = int(top2[1])
    top1_value = float(logits[top1])
    top2_value = float(logits[top2_id])
    return top1, top1_value, top2_id, top2_value, top1_value - top2_value


def compare_logits(hf: np.ndarray, cpp: np.ndarray, top_k: int) -> dict[str, Any]:
    if hf.shape != cpp.shape:
        raise ValueError(f"shape mismatch HF={hf.shape} C++={cpp.shape}")
    diff = cpp - hf
    hf_center = hf - np.mean(hf)
    cpp_center = cpp - np.mean(cpp)
    top_hf = np.argsort(hf)[-top_k:][::-1]
    top_cpp = np.argsort(cpp)[-top_k:][::-1]
    overlap = len(set(int(x) for x in top_hf).intersection(int(x) for x in top_cpp))
    hf_top1, hf_top1_logit, hf_top2, hf_top2_logit, hf_top1_top2_margin = top2_info(hf)
    cpp_top1, cpp_top1_logit, cpp_top2, cpp_top2_logit, cpp_top1_top2_margin = top2_info(cpp)
    hf_gap_to_cpp_argmax = float(hf[hf_top1] - hf[cpp_top1])
    cpp_gap_to_hf_argmax = float(cpp[cpp_top1] - cpp[hf_top1])
    return {
        "raw_rmse": float(np.sqrt(np.mean(diff * diff))),
        "centered_rmse": float(np.sqrt(np.mean((cpp_center - hf_center) ** 2))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "kl_hf_cpp": logits_kl(hf, cpp),
        "argmax_hf": hf_top1,
        "argmax_cpp": cpp_top1,
        "argmax_equal": bool(hf_top1 == cpp_top1),
        "hf_top1_logit": hf_top1_logit,
        "hf_top2": hf_top2,
        "hf_top2_logit": hf_top2_logit,
        "hf_top1_top2_margin": hf_top1_top2_margin,
        "cpp_top1_logit": cpp_top1_logit,
        "cpp_top2": cpp_top2,
        "cpp_top2_logit": cpp_top2_logit,
        "cpp_top1_top2_margin": cpp_top1_top2_margin,
        "hf_gap_to_cpp_argmax": hf_gap_to_cpp_argmax,
        "cpp_gap_to_hf_argmax": cpp_gap_to_hf_argmax,
        "top_k": top_k,
        "top_k_overlap": overlap,
        "top_k_overlap_ratio": overlap / top_k,
    }


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.preproc_baseline_dir = args.preproc_baseline_dir.resolve()
    args.hf_output_dir = args.hf_output_dir.resolve()
    args.cpp_output_dir = args.cpp_output_dir.resolve()

    samples = load_samples(args.preproc_baseline_dir, args.sample)

    if not args.skip_hf_run:
        run_hf_logits(args, samples)

    if not args.skip_cpp_run:
        for sample in samples:
            hf_manifest, _ = load_hf_sample(args.hf_output_dir, sample["label"])
            run_cpp_logits(args, sample, hf_manifest["forced_tokens"])

    for sample in samples:
        label = sample["label"]
        hf_manifest, hf_logits = load_hf_sample(args.hf_output_dir, label)
        cpp_manifest, cpp_logits = load_cpp_sample(args.cpp_output_dir, label)
        if len(hf_logits) != len(cpp_logits):
            raise ValueError(f"{label}: HF/C++ step count mismatch: {len(hf_logits)} != {len(cpp_logits)}")

        print(label)
        print(f"  input_ids_shape={hf_manifest['input_ids_shape']} forced_tokens={hf_manifest['forced_tokens']}")
        print(f"  cpp_n_past={cpp_manifest['n_past']}")
        mismatches: list[tuple[int, dict[str, Any]]] = []
        for i, (hf, cpp) in enumerate(zip(hf_logits, cpp_logits)):
            metrics = compare_logits(hf, cpp, args.top_k)
            if not metrics["argmax_equal"]:
                mismatches.append((i, metrics))
            print(
                f"  step={i:03d} raw_rmse={metrics['raw_rmse']:.6g} "
                f"centered_rmse={metrics['centered_rmse']:.6g} kl={metrics['kl_hf_cpp']:.6g} "
                f"argmax={metrics['argmax_hf']}/{metrics['argmax_cpp']} "
                f"argmax_equal={metrics['argmax_equal']} "
                f"top{args.top_k}_overlap={metrics['top_k_overlap']}/{args.top_k}"
            )
        print(f"  argmax_matches={len(hf_logits) - len(mismatches)}/{len(hf_logits)} mismatches={len(mismatches)}")
        if mismatches:
            near_tie_mismatches: list[tuple[int, dict[str, Any]]] = []
            non_tie_mismatches: list[tuple[int, dict[str, Any]]] = []
            for i, metrics in mismatches:
                near_tie = (
                    metrics["hf_gap_to_cpp_argmax"] <= args.tie_margin_threshold
                    or metrics["cpp_gap_to_hf_argmax"] <= args.tie_margin_threshold
                )
                (near_tie_mismatches if near_tie else non_tie_mismatches).append((i, metrics))

            print(
                f"  near_tie_mismatches={len(near_tie_mismatches)} "
                f"non_tie_mismatches={len(non_tie_mismatches)} "
                f"tie_margin_threshold={args.tie_margin_threshold:g}"
            )
            print("  mismatch_margins:")
            for i, metrics in mismatches:
                near_tie = (
                    metrics["hf_gap_to_cpp_argmax"] <= args.tie_margin_threshold
                    or metrics["cpp_gap_to_hf_argmax"] <= args.tie_margin_threshold
                )
                print(
                    f"    step={i:03d} hf={metrics['argmax_hf']} cpp={metrics['argmax_cpp']} "
                    f"hf_top1_top2_margin={metrics['hf_top1_top2_margin']:.6g} "
                    f"cpp_top1_top2_margin={metrics['cpp_top1_top2_margin']:.6g} "
                    f"hf_gap_to_cpp={metrics['hf_gap_to_cpp_argmax']:.6g} "
                    f"cpp_gap_to_hf={metrics['cpp_gap_to_hf_argmax']:.6g} "
                    f"near_tie={near_tie}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
