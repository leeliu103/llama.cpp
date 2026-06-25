#!/usr/bin/env python3
"""Generate and compare Phi-4-MM SigLIP hidden_states[-2] F32 baselines."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_MODEL_PATH = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct")
DEFAULT_PREPROC_BASELINE_DIR = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct-hf-baseline")
DEFAULT_TEXT_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/phi4mm-text-bf16.gguf")
DEFAULT_MMPROJ_GGUF = Path("/mnt/nas_share/models/gguf/phi4mm/mmproj-phi4mm-bf16.gguf")
DEFAULT_HF_F32_DIR = Path("/tmp/phi4mm_hf_f32_hidden_baseline")
DEFAULT_CPP_DIR = Path("/tmp/phi4mm_cpp_hidden_parity")

VISION_TOWER_PREFIX = "model.embed_tokens_extend.image_embed.img_processor."
PHI4MM_EXPORTED_VISION_LAYERS = 26
DIAGNOSTIC_DETAIL_BASE_NAMES = (
    "layer_inp_normed",
    "Qcur",
    "Kcur",
    "Vcur",
    "attn_scores_raw",
    "attn_probs",
    "kqv_out",
    "attn_out",
    "ffn_inp",
    "ffn_inp_normed",
    "ffn_out",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--preproc-baseline-dir",
        type=Path,
        default=DEFAULT_PREPROC_BASELINE_DIR,
        help=(
            "Existing HF baseline directory used for sample metadata and verified "
            "input_image_embeds/image_attention_mask tensors. This can remain the BF16 baseline."
        ),
    )
    parser.add_argument("--hf-f32-dir", type=Path, default=DEFAULT_HF_F32_DIR)
    parser.add_argument("--cpp-output-dir", type=Path, default=DEFAULT_CPP_DIR)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--model", type=Path, default=DEFAULT_TEXT_GGUF)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ_GGUF)
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Sample label to compare. Defaults to all samples in the preproc baseline manifest.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for HF F32 generation: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--skip-hf-run",
        action="store_true",
        help="Reuse existing HF F32 hidden_states_minus2.pt files in --hf-f32-dir.",
    )
    parser.add_argument(
        "--skip-cpp-run",
        action="store_true",
        help="Reuse existing C++ dumps in --cpp-output-dir.",
    )
    parser.add_argument(
        "--max-abs-threshold",
        type=float,
        default=None,
        help="If set, fail when C++ vs HF F32 max_abs exceeds this value.",
    )
    parser.add_argument(
        "--dump-layer-diagnostics",
        action="store_true",
        help=(
            "Also dump/compare pos_embed, every exported layer_out, and selected op-region tensors "
            "for --diagnostic-layer."
        ),
    )
    parser.add_argument(
        "--diagnostic-layer",
        type=int,
        default=0,
        help="Layer index for detailed op-region diagnostics. Default: 0.",
    )
    parser.add_argument(
        "--diagnostic-threshold",
        type=float,
        default=1e-3,
        help="Threshold used to report the first diagnostic tensor exceeding max_abs. Default: 1e-3.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def configure_determinism() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "ieee"
        elif hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = "ieee"
        else:
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"torch": torch.__version__}
    for name in ("transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def load_samples(preproc_baseline_dir: Path, requested: list[str]) -> list[dict[str, Any]]:
    manifest_path = preproc_baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if requested:
        wanted = set(requested)
        samples = [sample for sample in samples if sample["label"] in wanted]
        missing = sorted(wanted - {sample["label"] for sample in samples})
        if missing:
            raise ValueError(f"Requested sample(s) not found in {manifest_path}: {', '.join(missing)}")
    return samples


def import_official_siglip(model_path: Path) -> Any:
    module_path = model_path / "vision_siglip_navit.py"
    spec = importlib.util.spec_from_file_location("phi4mm_official_vision_siglip_navit", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_official_vision_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    module = import_official_siglip(model_path)
    vision_model = module.get_siglip_vision_model(_flash_attn_2_enabled=False)

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, file_name in index["weight_map"].items():
        if key.startswith(VISION_TOWER_PREFIX):
            keys_by_file[file_name].append(key)

    if not keys_by_file:
        raise RuntimeError(f"No vision tower tensors with prefix {VISION_TOWER_PREFIX!r} found in {index_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for file_name, keys in sorted(keys_by_file.items()):
        with safe_open(model_path / file_name, framework="pt", device="cpu") as handle:
            for key in sorted(keys):
                state_dict[key[len(VISION_TOWER_PREFIX):]] = handle.get_tensor(key).to(torch.float32)

    missing, unexpected = vision_model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Vision tower load mismatch: missing={missing}, unexpected={unexpected}")

    vision_model.eval()
    vision_model.to(device=device, dtype=torch.float32)
    return vision_model


def safe_file_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)


def diagnostic_tensor_names(layer: int) -> list[str]:
    names = ["pos_embed"]
    names.extend(f"layer_out-{il}" for il in range(PHI4MM_EXPORTED_VISION_LAYERS))
    names.extend(f"{base}-{layer}" for base in DIAGNOSTIC_DETAIL_BASE_NAMES)
    return names


def save_hf_diagnostics(
    *,
    sample_dir: Path,
    hidden_states: tuple[torch.Tensor, ...],
    records: dict[str, torch.Tensor],
    diagnostic_layer: int,
) -> None:
    diag_dir = sample_dir / "hf_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, dict[str, Any]] = {}

    def save_one(name: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().cpu().to(torch.float32)
        file_name = safe_file_stem(name) + ".pt"
        torch.save(value, diag_dir / file_name)
        saved[name] = {"file": file_name, **save_tensor_stats(value)}

    save_one("pos_embed", hidden_states[0])
    for il in range(PHI4MM_EXPORTED_VISION_LAYERS):
        save_one(f"layer_out-{il}", hidden_states[il + 1])

    for base_name in DIAGNOSTIC_DETAIL_BASE_NAMES:
        name = f"{base_name}-{diagnostic_layer}"
        if name in records:
            save_one(name, records[name])

    manifest = {
        "created_at_utc": utc_now(),
        "diagnostic_layer": diagnostic_layer,
        "tensors": saved,
    }
    (diag_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def save_tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().cpu()
    info: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        return info
    values = t.float()
    info.update(
        {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        }
    )
    return info


def generate_hf_f32_baseline(
    *,
    model: torch.nn.Module,
    device: torch.device,
    samples: list[dict[str, Any]],
    preproc_baseline_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    root_manifest: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "model_path": str(args.model_path),
        "preproc_baseline_dir": str(preproc_baseline_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": "float32",
        "vision_tower_prefix": VISION_TOWER_PREFIX,
        "package_versions": package_versions(),
        "samples": [],
        "notes": [
            "Generated from the official Microsoft SigLIP vision tower safetensors.",
            "input_image_embeds and image_attention_mask are read from the existing HF preprocessing baseline.",
        ],
    }

    for sample in samples:
        label = sample["label"]
        source_dir = preproc_baseline_dir / label
        sample_dir = output_dir / label
        sample_dir.mkdir(parents=True, exist_ok=True)

        img_embeds = torch.load(source_dir / "input_image_embeds.pt", map_location="cpu").to(torch.float32)
        image_attention_mask = torch.load(source_dir / "image_attention_mask.pt", map_location="cpu").to(torch.bool)
        flat_embeds = img_embeds.flatten(0, 1).to(device=device, dtype=torch.float32)
        flat_mask = image_attention_mask.flatten(0, 1).to(device=device)

        records: dict[str, torch.Tensor] = {}
        handles: list[Any] = []
        if args.dump_layer_diagnostics:
            if args.diagnostic_layer < 0 or args.diagnostic_layer >= PHI4MM_EXPORTED_VISION_LAYERS:
                raise ValueError(
                    f"--diagnostic-layer must be in [0, {PHI4MM_EXPORTED_VISION_LAYERS - 1}], "
                    f"got {args.diagnostic_layer}"
                )

            def capture(name: str, value: Any) -> None:
                if isinstance(value, (tuple, list)):
                    value = value[0]
                if torch.is_tensor(value):
                    records[name] = value.detach().cpu().to(torch.float32)

            def capture_heads(name: str, value: Any) -> None:
                if isinstance(value, (tuple, list)):
                    value = value[0]
                if not torch.is_tensor(value):
                    return
                batch_size, seq_len, embed_dim = value.shape
                num_heads = layer.self_attn.num_heads
                head_dim = embed_dim // num_heads
                records[name] = value.detach().view(batch_size, seq_len, num_heads, head_dim).cpu().to(torch.float32)

            layer = model.encoder.layers[args.diagnostic_layer]
            handles.append(
                layer.layer_norm1.register_forward_hook(
                    lambda _module, _inputs, output: capture(f"layer_inp_normed-{args.diagnostic_layer}", output)
                )
            )
            handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture_heads(f"Qcur-{args.diagnostic_layer}", output)
                )
            )
            handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture_heads(f"Kcur-{args.diagnostic_layer}", output)
                )
            )
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture_heads(f"Vcur-{args.diagnostic_layer}", output)
                )
            )
            handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(
                    lambda _module, inputs: capture(f"kqv_out-{args.diagnostic_layer}", inputs[0] if inputs else None)
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    lambda _module, _inputs, output: capture(f"attn_out-{args.diagnostic_layer}", output)
                )
            )
            handles.append(
                layer.layer_norm2.register_forward_pre_hook(
                    lambda _module, inputs: capture(f"ffn_inp-{args.diagnostic_layer}", inputs[0] if inputs else None)
                )
            )
            handles.append(
                layer.layer_norm2.register_forward_hook(
                    lambda _module, _inputs, output: capture(f"ffn_inp_normed-{args.diagnostic_layer}", output)
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(
                    lambda _module, _inputs, output: capture(f"ffn_out-{args.diagnostic_layer}", output)
                )
            )

        try:
            with torch.inference_mode():
                outputs = model(
                    flat_embeds,
                    patch_attention_mask=flat_mask,
                    output_attentions=args.dump_layer_diagnostics,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        hidden = outputs.hidden_states[-2].detach().cpu().to(torch.float32)
        if args.dump_layer_diagnostics:
            layer_name = args.diagnostic_layer
            q = records.get(f"Qcur-{layer_name}")
            k = records.get(f"Kcur-{layer_name}")
            if q is not None and k is not None:
                qh = q.permute(0, 2, 1, 3)
                kh = k.permute(0, 2, 1, 3)
                records[f"attn_scores_raw-{layer_name}"] = torch.matmul(qh, kh.transpose(2, 3)).cpu().to(torch.float32)
            if outputs.attentions is not None and len(outputs.attentions) > args.diagnostic_layer:
                records[f"attn_probs-{layer_name}"] = outputs.attentions[args.diagnostic_layer].detach().cpu().to(torch.float32)
        torch.save(hidden, sample_dir / "hidden_states_minus2.pt")
        if args.dump_layer_diagnostics:
            save_hf_diagnostics(
                sample_dir=sample_dir,
                hidden_states=outputs.hidden_states,
                records=records,
                diagnostic_layer=args.diagnostic_layer,
            )

        sample_manifest = {
            "label": label,
            "image_path": sample.get("image_path"),
            "source_manifest": str(source_dir / "manifest.json"),
            "created_at_utc": utc_now(),
            "artifacts": {
                "hidden_states_minus2": {
                    "file": "hidden_states_minus2.pt",
                    **save_tensor_stats(hidden),
                    "note": "Official SigLIP hidden_states[-2] generated in torch.float32.",
                },
            },
        }
        (sample_dir / "manifest.json").write_text(json.dumps(sample_manifest, indent=2, sort_keys=True), encoding="utf-8")
        root_manifest["samples"].append(
            {
                "label": label,
                "image_path": sample.get("image_path"),
                "manifest": f"{label}/manifest.json",
                "artifact_names": ["hidden_states_minus2"],
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(root_manifest, indent=2, sort_keys=True), encoding="utf-8")


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
        "--flash-attn",
        "off",
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
    env["MTMD_DEBUG_PHI4MM_HIDDEN_STATES_DUMP"] = str(sample_dir)
    if args.dump_layer_diagnostics:
        layer_dump_dir = sample_dir / "cpp_diagnostics"
        layer_dump_dir.mkdir(parents=True, exist_ok=True)
        env["MTMD_DEBUG_PHI4MM_LAYER_DUMP"] = str(layer_dump_dir)
        env["MTMD_DEBUG_PHI4MM_LAYER_DUMP_NAMES"] = ",".join(diagnostic_tensor_names(args.diagnostic_layer))
        env["MTMD_DEBUG_PHI4MM_LAYER_DUMP_N_PATCHES"] = "1024"
    subprocess.run(cmd, check=True, env=env)


def load_cpp_hidden(sample_dir: Path) -> torch.Tensor:
    manifest_path = sample_dir / "hidden_states_minus2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info = manifest["hidden_states_minus2"]
    data = np.fromfile(sample_dir / info.get("file", "hidden_states_minus2.f32"), dtype=np.float32)
    expected = int(np.prod(info["shape"]))
    if data.size != expected:
        raise ValueError(f"{sample_dir}: got {data.size} values, expected {expected} for shape {info['shape']}")
    return torch.from_numpy(data.reshape(info["shape"]))


def compare_hidden(hf_f32_dir: Path, cpp_dir: Path, sample: dict[str, Any]) -> dict[str, Any]:
    label = sample["label"]
    hf = torch.load(hf_f32_dir / label / "hidden_states_minus2.pt", map_location="cpu").to(torch.float32)
    cpp = load_cpp_hidden(cpp_dir / label).to(torch.float32)
    if tuple(hf.shape) != tuple(cpp.shape):
        raise ValueError(f"{label}: shape mismatch HF={tuple(hf.shape)} C++={tuple(cpp.shape)}")
    diff = (cpp - hf).abs()
    return {
        "shape": list(hf.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def load_cpp_diagnostic_tensor(sample_dir: Path, name: str) -> torch.Tensor:
    diag_dir = sample_dir / "cpp_diagnostics"
    manifest = json.loads((diag_dir / "phi4mm_layer_dumps.json").read_text(encoding="utf-8"))
    info = manifest["tensors"][name]
    data = np.fromfile(diag_dir / info["file"], dtype=np.float32)
    expected = int(np.prod(info["shape"]))
    if data.size != expected:
        raise ValueError(f"{diag_dir / info['file']}: got {data.size} values, expected {expected}")
    return torch.from_numpy(data.reshape(info["shape"]))


def load_hf_diagnostic_tensor(sample_dir: Path, name: str) -> torch.Tensor:
    diag_dir = sample_dir / "hf_diagnostics"
    manifest = json.loads((diag_dir / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["tensors"][name]
    return torch.load(diag_dir / info["file"], map_location="cpu").to(torch.float32)


def compare_diagnostics(
    hf_f32_dir: Path,
    cpp_dir: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[str, dict[str, Any]]]:
    label = sample["label"]
    results: list[tuple[str, dict[str, Any]]] = []
    for name in diagnostic_tensor_names(args.diagnostic_layer):
        hf = load_hf_diagnostic_tensor(hf_f32_dir / label, name)
        cpp = load_cpp_diagnostic_tensor(cpp_dir / label, name).to(torch.float32)
        if hf.ndim == cpp.ndim + 1 and hf.shape[0] == 1 and tuple(hf.shape[1:]) == tuple(cpp.shape):
            cpp = cpp.unsqueeze(0)
        if tuple(hf.shape) != tuple(cpp.shape):
            raise ValueError(f"{label} {name}: shape mismatch HF={tuple(hf.shape)} C++={tuple(cpp.shape)}")
        diff = (cpp - hf).abs()
        results.append(
            (
                name,
                {
                    "shape": list(hf.shape),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                },
            )
        )
    return results


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.preproc_baseline_dir = args.preproc_baseline_dir.resolve()
    args.hf_f32_dir = args.hf_f32_dir.resolve()
    args.cpp_output_dir = args.cpp_output_dir.resolve()

    configure_determinism()
    device = select_device(args.device)
    samples = load_samples(args.preproc_baseline_dir, args.sample)

    if not args.skip_hf_run:
        print(f"Generating HF F32 hidden baseline in {args.hf_f32_dir}", file=sys.stderr)
        print(f"HF device={device}", file=sys.stderr)
        model = load_official_vision_model(args.model_path, device)
        generate_hf_f32_baseline(
            model=model,
            device=device,
            samples=samples,
            preproc_baseline_dir=args.preproc_baseline_dir,
            output_dir=args.hf_f32_dir,
            args=args,
        )

    if not args.skip_cpp_run:
        for sample in samples:
            run_cpp_dump(args, sample, args.cpp_output_dir / sample["label"])

    failed = False
    all_results: dict[str, dict[str, Any]] = {}
    all_diagnostics: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for sample in samples:
        result = compare_hidden(args.hf_f32_dir, args.cpp_output_dir, sample)
        all_results[sample["label"]] = result
        if args.max_abs_threshold is not None and result["max_abs"] > args.max_abs_threshold:
            failed = True
        if args.dump_layer_diagnostics:
            all_diagnostics[sample["label"]] = compare_diagnostics(args.hf_f32_dir, args.cpp_output_dir, sample, args)

    for label, result in all_results.items():
        print(label)
        print(
            f"  hidden_states_minus2: shape={result['shape']} "
            f"max_abs={result['max_abs']:.9g} mean_abs={result['mean_abs']:.9g}"
        )
        if args.dump_layer_diagnostics:
            first_exceeding: tuple[str, dict[str, Any]] | None = None
            for name, diag_result in all_diagnostics[label]:
                if first_exceeding is None and diag_result["max_abs"] > args.diagnostic_threshold:
                    first_exceeding = (name, diag_result)
                print(
                    f"  diag {name}: shape={diag_result['shape']} "
                    f"max_abs={diag_result['max_abs']:.9g} mean_abs={diag_result['mean_abs']:.9g}"
                )
            if first_exceeding is None:
                print(f"  first_diag_exceeding_{args.diagnostic_threshold:g}: none")
            else:
                name, diag_result = first_exceeding
                print(
                    f"  first_diag_exceeding_{args.diagnostic_threshold:g}: {name} "
                    f"max_abs={diag_result['max_abs']:.9g} mean_abs={diag_result['mean_abs']:.9g}"
                )

    if failed:
        print(f"FAILED: max_abs exceeded threshold {args.max_abs_threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
