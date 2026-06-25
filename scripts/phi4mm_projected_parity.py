#!/usr/bin/env python3
"""Compare Phi-4-MM projected image embeddings against an HF F32 reference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_MODEL_PATH = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct")
DEFAULT_PREPROC_BASELINE_DIR = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct-hf-baseline")
DEFAULT_HF_HIDDEN_DIR = Path("/tmp/phi4mm_hf_f32_masked_attn_diag_f32mmproj_l0")
DEFAULT_HF_PROJECTED_DIR = Path("/tmp/phi4mm_hf_f32_projected")
DEFAULT_CPP_DIR = Path("/tmp/phi4mm_cpp_projected")
DEFAULT_TEXT_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/phi4mm-text-bf16.gguf")
DEFAULT_MMPROJ_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/mmproj-phi4mm-f32.gguf")

IMAGE_PREFIX = "model.embed_tokens_extend.image_embed."
PROJECTOR_KEYS = {
    "glb_GN",
    "sub_GN",
    "img_projection.0.weight",
    "img_projection.0.bias",
    "img_projection.2.weight",
    "img_projection.2.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preproc-baseline-dir", type=Path, default=DEFAULT_PREPROC_BASELINE_DIR)
    parser.add_argument("--hf-hidden-dir", type=Path, default=DEFAULT_HF_HIDDEN_DIR)
    parser.add_argument("--hf-projected-dir", type=Path, default=DEFAULT_HF_PROJECTED_DIR)
    parser.add_argument("--cpp-output-dir", type=Path, default=DEFAULT_CPP_DIR)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--model", type=Path, default=DEFAULT_TEXT_GGUF)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ_GGUF)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--skip-hf-run", action="store_true")
    parser.add_argument("--skip-cpp-run", action="store_true")
    parser.add_argument("--max-rmse-threshold", type=float, default=None)
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


def load_projector_tensors(model_path: Path) -> dict[str, torch.Tensor]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    wanted = {IMAGE_PREFIX + key for key in PROJECTOR_KEYS}
    for key, file_name in index["weight_map"].items():
        if key in wanted:
            keys_by_file[file_name].append(key)

    found: dict[str, torch.Tensor] = {}
    for file_name, keys in sorted(keys_by_file.items()):
        with safe_open(model_path / file_name, framework="pt", device="cpu") as handle:
            for key in keys:
                found[key.removeprefix(IMAGE_PREFIX)] = handle.get_tensor(key).to(torch.float32)

    missing = sorted(PROJECTOR_KEYS - set(found))
    if missing:
        raise RuntimeError(f"Missing projector tensors: {', '.join(missing)}")
    return found


def build_projected_embeddings(
    hidden_states: torch.Tensor,
    image_sizes: torch.Tensor,
    image_attention_mask: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    image_attention_mask = image_attention_mask.to(torch.bool)

    n_crops, n_tokens, n_embd = hidden_states.shape
    if n_tokens != 32 * 32 or n_embd != 1152:
        raise ValueError(f"Unexpected hidden_states shape: {tuple(hidden_states.shape)}")

    pooled = hidden_states.view(n_crops, 32, 32, n_embd).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(pooled, kernel_size=2, stride=2)
    pooled = pooled.permute(0, 2, 3, 1).contiguous().view(n_crops, 16 * 16, n_embd)

    glb_gn = tensors["glb_GN"]
    sub_gn = tensors["sub_GN"]
    w0 = tensors["img_projection.0.weight"]
    b0 = tensors["img_projection.0.bias"]
    w2 = tensors["img_projection.2.weight"]
    b2 = tensors["img_projection.2.bias"]

    out_per_image: list[torch.Tensor] = []
    for sample_idx in range(image_sizes.shape[0]):
        height, width = image_sizes[sample_idx].tolist()
        grid_y = int(height) // 448
        grid_x = int(width) // 448
        n_sub_crops = grid_y * grid_x

        crop_features = pooled[: 1 + n_sub_crops]
        global_img_feature = crop_features[:1]

        global_img = global_img_feature.reshape(1, 16, 16, n_embd)
        global_newlines = sub_gn.repeat(1, 16, 1, 1)
        global_img = torch.cat([global_img, global_newlines], dim=2).reshape(1, -1, n_embd)

        sub_img = crop_features[1 : 1 + n_sub_crops]
        sub_img = sub_img.reshape(n_sub_crops, 16, 16, n_embd)
        sub_img = sub_img.reshape(1, grid_y, grid_x, 16, 16, n_embd)
        sub_img = sub_img.permute(0, 1, 3, 2, 4, 5).reshape(1, grid_y * 16, grid_x * 16, n_embd)

        mask = image_attention_mask[sample_idx, 1 : n_sub_crops + 1, 0::2, 0::2]
        mask = mask.reshape(1, grid_y, grid_x, 16, 16).permute(0, 1, 3, 2, 4)
        mask = mask.reshape(1, grid_y * 16, grid_x * 16)
        useful_height = int(mask[0, :, 0].sum().item())
        useful_width = int(mask[0, 0, :].sum().item())

        sub_img = sub_img[:, :useful_height, :useful_width]
        sub_newlines = sub_gn.repeat(1, useful_height, 1, 1)
        sub_img = torch.cat([sub_img, sub_newlines], dim=2).reshape(1, -1, n_embd)

        assembled = torch.cat([sub_img, glb_gn, global_img], dim=1)
        projected = F.linear(F.gelu(F.linear(assembled, w0, b0), approximate="none"), w2, b2)
        out_per_image.append(projected)

    return torch.cat(out_per_image, dim=1)


def generate_hf_projected(
    *,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
) -> None:
    args.hf_projected_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        label = sample["label"]
        source_dir = args.preproc_baseline_dir / label
        hidden_dir = args.hf_hidden_dir / label
        sample_dir = args.hf_projected_dir / label
        sample_dir.mkdir(parents=True, exist_ok=True)

        hidden = torch.load(hidden_dir / "hidden_states_minus2.pt", map_location="cpu").to(torch.float32)
        image_sizes = torch.load(source_dir / "image_sizes.pt", map_location="cpu").to(torch.long)
        image_attention_mask = torch.load(source_dir / "image_attention_mask.pt", map_location="cpu")
        projected = build_projected_embeddings(hidden, image_sizes, image_attention_mask, tensors).to(torch.float32)
        torch.save(projected, sample_dir / "projected_embeddings.pt")
        (sample_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "artifact": {
                        "file": "projected_embeddings.pt",
                        "dtype": "float32",
                        "shape": list(projected.shape),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def run_cpp_dump(args: argparse.Namespace, sample: dict[str, Any], sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    debug_bin = args.build_dir / "bin" / "llama-mtmd-debug"
    cmd = [
        str(debug_bin),
        "-m", str(args.model),
        "--mmproj", str(args.mmproj),
        "--no-warmup",
        "--no-mmproj-offload",
        "--flash-attn", "off",
        "-ngl", "0",
        "-p", "preproc",
        "-n", "1",
        "--image", str(sample["image_path"]),
    ]
    env = os.environ.copy()
    env["MTMD_DEBUG_PHI4MM_PROJECTED_EMBEDDINGS_DUMP"] = str(sample_dir)
    subprocess.run(cmd, check=True, env=env)


def load_cpp_projected(sample_dir: Path) -> torch.Tensor:
    manifest = json.loads((sample_dir / "projected_embeddings.json").read_text(encoding="utf-8"))
    info = manifest["projected_embeddings"]
    data = np.fromfile(sample_dir / info["file"], dtype=np.float32)
    expected = int(np.prod(info["shape"]))
    if data.size != expected:
        raise ValueError(f"{sample_dir}: got {data.size} values, expected {expected}")
    return torch.from_numpy(data.reshape(info["shape"])).unsqueeze(0)


def compare_one(args: argparse.Namespace, sample: dict[str, Any]) -> dict[str, Any]:
    label = sample["label"]
    hf = torch.load(args.hf_projected_dir / label / "projected_embeddings.pt", map_location="cpu").to(torch.float32)
    cpp = load_cpp_projected(args.cpp_output_dir / label).to(torch.float32)
    if tuple(hf.shape) != tuple(cpp.shape):
        raise ValueError(f"{label}: shape mismatch HF={tuple(hf.shape)} C++={tuple(cpp.shape)}")
    diff = cpp - hf
    return {
        "shape": list(hf.shape),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.preproc_baseline_dir = args.preproc_baseline_dir.resolve()
    args.hf_hidden_dir = args.hf_hidden_dir.resolve()
    args.hf_projected_dir = args.hf_projected_dir.resolve()
    args.cpp_output_dir = args.cpp_output_dir.resolve()

    samples = load_samples(args.preproc_baseline_dir, args.sample)
    if not args.skip_hf_run:
        tensors = load_projector_tensors(args.model_path)
        generate_hf_projected(samples=samples, args=args, tensors=tensors)

    if not args.skip_cpp_run:
        for sample in samples:
            run_cpp_dump(args, sample, args.cpp_output_dir / sample["label"])

    failed = False
    for sample in samples:
        result = compare_one(args, sample)
        print(sample["label"])
        print(
            f"  projected_embeddings: shape={result['shape']} "
            f"max_abs={result['max_abs']:.9g} mean_abs={result['mean_abs']:.9g} rmse={result['rmse']:.9g}"
        )
        if args.max_rmse_threshold is not None and result["rmse"] > args.max_rmse_threshold:
            failed = True

    if failed:
        print(f"FAILED: rmse exceeded threshold {args.max_rmse_threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
