#!/usr/bin/env python3
"""Compare llama.cpp Phi-4-MM preprocessing dumps against HF baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_BASELINE_DIR = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct-hf-baseline")
DEFAULT_TEXT_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/phi4mm-text-bf16.gguf")
DEFAULT_MMPROJ_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/mmproj-phi4mm-bf16.gguf")


PREPROC_ARTIFACTS = {
    "input_image_embeds": ("input_image_embeds.f32", np.float32),
    "image_attention_mask": ("image_attention_mask.u8", np.uint8),
    "image_sizes": ("image_sizes.i64", np.int64),
    "num_img_tokens": ("num_img_tokens.i64", np.int64),
}
HIDDEN_STATE_ARTIFACTS = {
    "hidden_states_minus2": ("hidden_states_minus2.f32", np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_TEXT_GGUF)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ_GGUF)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phi4mm_cpp_preproc_parity"))
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Baseline sample label to compare. Defaults to all samples in the baseline manifest.",
    )
    parser.add_argument(
        "--max-abs-threshold",
        type=float,
        default=None,
        help="If set, fail when any floating artifact exceeds this max_abs delta.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Reuse existing C++ dumps in --output-dir instead of invoking llama-mtmd-debug.",
    )
    parser.add_argument(
        "--include-hidden-states",
        action="store_true",
        help="Also run the Phi-4-MM SigLIP hidden_states[-2] graph and compare hidden_states_minus2.",
    )
    return parser.parse_args()


def selected_artifacts(include_hidden_states: bool) -> dict[str, tuple[str, np.dtype]]:
    artifacts = dict(PREPROC_ARTIFACTS)
    if include_hidden_states:
        artifacts.update(HIDDEN_STATE_ARTIFACTS)
    return artifacts


def load_baseline_samples(baseline_dir: Path, requested: list[str]) -> list[dict[str, Any]]:
    manifest_path = baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if requested:
        wanted = set(requested)
        samples = [sample for sample in samples if sample["label"] in wanted]
        missing = sorted(wanted - {sample["label"] for sample in samples})
        if missing:
            raise ValueError(f"Requested sample(s) not found in {manifest_path}: {', '.join(missing)}")
    return samples


def run_cpp_dump(args: argparse.Namespace, sample: dict[str, Any], sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    debug_bin = args.build_dir / "bin" / "llama-mtmd-debug"
    image_path = Path(sample["image_path"])
    cmd = [
        str(debug_bin),
        "-m",
        str(args.model),
        "--mmproj",
        str(args.mmproj),
        "--no-warmup",
        "--no-mmproj-offload",
        "-ngl",
        "0",
        "-p",
        "preproc",
        "-n",
        "1",
        "--image",
        str(image_path),
    ]
    env = os.environ.copy()
    env["MTMD_DEBUG_PREPROC_DUMP"] = str(sample_dir)
    if args.include_hidden_states:
        env["MTMD_DEBUG_PHI4MM_HIDDEN_STATES_DUMP"] = str(sample_dir)
    subprocess.run(cmd, check=True, env=env)


def load_cpp_artifact(
    sample_dir: Path,
    manifest: dict[str, Any],
    name: str,
    artifacts: dict[str, tuple[str, np.dtype]],
) -> torch.Tensor:
    if name in manifest:
        info = manifest[name]
    elif name in HIDDEN_STATE_ARTIFACTS:
        hidden_manifest_path = sample_dir / "hidden_states_minus2.json"
        hidden_manifest = json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
        info = hidden_manifest[name]
    else:
        raise ValueError(f"C++ dump for {sample_dir.name} is missing {name}")

    default_file_name, dtype = artifacts[name]
    path = sample_dir / info.get("file", default_file_name)
    data = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(info["shape"]))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} values, expected {expected} for shape {info['shape']}")
    return torch.from_numpy(data.reshape(info["shape"]))


def compare_artifact(
    baseline_sample_dir: Path,
    cpp_sample_dir: Path,
    name: str,
    artifacts: dict[str, tuple[str, np.dtype]],
) -> dict[str, Any]:
    cpp_manifest = json.loads((cpp_sample_dir / "manifest.json").read_text(encoding="utf-8"))
    if name not in cpp_manifest and name not in HIDDEN_STATE_ARTIFACTS:
        raise ValueError(f"C++ dump for {cpp_sample_dir.name} is missing {name}")

    hf = torch.load(baseline_sample_dir / f"{name}.pt", map_location="cpu")
    cpp = load_cpp_artifact(cpp_sample_dir, cpp_manifest, name, artifacts)

    if tuple(hf.shape) != tuple(cpp.shape):
        raise ValueError(f"{cpp_sample_dir.name} {name}: shape mismatch HF={tuple(hf.shape)} C++={tuple(cpp.shape)}")

    if hf.is_floating_point() or cpp.is_floating_point():
        diff = (cpp.to(torch.float32) - hf.to(torch.float32)).abs()
        return {
            "shape": list(hf.shape),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }

    equal = bool(torch.equal(cpp, hf))
    max_abs = int((cpp.to(torch.int64) - hf.to(torch.int64)).abs().max().item())
    return {
        "shape": list(hf.shape),
        "equal": equal,
        "max_abs": max_abs,
        "mean_abs": float((cpp.to(torch.float32) - hf.to(torch.float32)).abs().mean().item()),
    }


def main() -> int:
    args = parse_args()
    samples = load_baseline_samples(args.baseline_dir, args.sample)
    artifacts = selected_artifacts(args.include_hidden_states)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    failed = False

    for sample in samples:
        label = sample["label"]
        cpp_sample_dir = args.output_dir / label
        if not args.skip_run:
            run_cpp_dump(args, sample, cpp_sample_dir)

        baseline_sample_dir = args.baseline_dir / label
        sample_results: dict[str, dict[str, Any]] = {}
        for name in artifacts:
            result = compare_artifact(baseline_sample_dir, cpp_sample_dir, name, artifacts)
            sample_results[name] = result
            if args.max_abs_threshold is not None and "max_abs" in result and result["max_abs"] > args.max_abs_threshold:
                failed = True
        all_results[label] = sample_results

    for label, results in all_results.items():
        print(label)
        for name, result in results.items():
            if "equal" in result:
                print(
                    f"  {name}: shape={result['shape']} equal={result['equal']} "
                    f"max_abs={result['max_abs']} mean_abs={result['mean_abs']:.9g}"
                )
            else:
                print(
                    f"  {name}: shape={result['shape']} "
                    f"max_abs={result['max_abs']:.9g} mean_abs={result['mean_abs']:.9g}"
                )

    if failed:
        print(f"FAILED: max_abs exceeded threshold {args.max_abs_threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
