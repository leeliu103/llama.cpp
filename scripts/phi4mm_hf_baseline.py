#!/usr/bin/env python3
"""Dump official Hugging Face Phi-4-multimodal vision baselines.

This harness is intentionally narrow: it loads the local Microsoft
Phi-4-multimodal-instruct remote code, runs one image at a time in VISION mode,
and captures the image-tower/projection tensors needed for llama.cpp parity
work.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import GenerationConfig
except Exception:  # pragma: no cover - optional for older local installs
    GenerationConfig = None


DEFAULT_MODEL_PATH = Path("/mnt/nas_share/models/microsoft/Phi-4-multimodal-instruct")
DEFAULT_IMAGES = (
    DEFAULT_MODEL_PATH / "figures" / "vision_radar.png",
    DEFAULT_MODEL_PATH / "figures" / "multi_image.png",
)
DEFAULT_PROMPT = "<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>"
IMAGE_TOKEN_RE = re.compile(r"(<\|image_\d+\|>|<\|endoftext10\|>)")
IMAGE_MODE_VISION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump Phi-4-multimodal-instruct HF vision tensors for parity work. "
            "By default, runs the two official figure images; pass "
            "--simple-object-image for the third plan image."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Additional image path to dump. May be passed more than once.",
    )
    parser.add_argument(
        "--simple-object-image",
        type=Path,
        default=None,
        help="Optional simple object image path for the plan's third baseline.",
    )
    parser.add_argument(
        "--no-default-images",
        action="store_true",
        help="Do not automatically include the official vision_radar.png and multi_image.png images.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=(
            "Prompt containing <|image_1|>, <|endoftext10|>, or a {image} placeholder. "
            f"Default: {DEFAULT_PROMPT!r}"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phi4mm_hf_baseline"))
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Model dtype passed to from_pretrained. Default: auto.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        help=(
            "Attention implementation passed to from_pretrained. "
            "Default eager avoids requiring a specific GPU/flash-attn path."
        ),
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow transformers to fetch missing files. Default is local-files-only.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-logits-to-keep",
        type=int,
        default=1,
        help="Forward-pass logits window. Use 1 for first generated token logits.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="If greater than zero, also run deterministic generation for this many tokens.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write manifests and tensor stats only; do not write .pt tensor payloads.",
    )
    parser.add_argument(
        "--skip-vision-hidden-states",
        action="store_true",
        help="Do not write the large SigLIP hidden_states[-2] .pt file.",
    )
    parser.add_argument(
        "--dump-decoder-hidden-minus2",
        action="store_true",
        help="Also dump decoder outputs.hidden_states[-2]. This can be very large.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run the official processor and dump input_image_embeds/image_sizes/image_attention_mask/num_img_tokens.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_dtype(dtype_arg: str) -> torch.dtype | str:
    if dtype_arg == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"torch": torch.__version__}
    for name in ("transformers", "peft", "tokenizers", "torchvision", "safetensors", "backoff", "flash_attn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def validate_prompt(prompt: str) -> str:
    prompt = prompt.replace("{image}", "<|image_1|>")
    if not IMAGE_TOKEN_RE.search(prompt):
        raise ValueError(
            "Prompt must contain <|image_1|>, <|endoftext10|>, or a {image} placeholder "
            "so the processor can expand image tokens."
        )
    return prompt


def safe_label(path: Path, used: set[str], preferred: str | None = None) -> str:
    base = preferred or path.stem
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or "image"
    candidate = label
    suffix = 2
    while candidate in used:
        candidate = f"{label}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def resolve_images(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    images: list[tuple[str, Path, str]] = []
    used: set[str] = set()

    if not args.no_default_images:
        for path in DEFAULT_IMAGES:
            images.append((safe_label(path, used), path, "official"))

    for path in args.image:
        images.append((safe_label(path, used), path, "extra"))

    if args.simple_object_image is not None:
        images.append((safe_label(args.simple_object_image, used, "simple_object"), args.simple_object_image, "simple_object"))

    if not images:
        raise ValueError("No images selected. Pass --image or omit --no-default-images.")

    missing = [str(path) for _, path, _ in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image path(s) not found: " + ", ".join(missing))

    return images


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().cpu()
    info: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        return info

    if t.is_complex():
        return info

    if t.dtype == torch.bool:
        info["true_count"] = int(t.sum().item())
        return info

    if torch.is_floating_point(t):
        finite = torch.isfinite(t)
        info["finite_count"] = int(finite.sum().item())
        if finite.any():
            values = t[finite].float()
            info.update(
                {
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
                }
            )
        return info

    info.update(
        {
            "min": int(t.min().item()),
            "max": int(t.max().item()),
            "sum": int(t.sum().item()),
        }
    )
    return info


def describe_value(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return tensor_stats(value)
    if isinstance(value, (list, tuple)) and all(torch.is_tensor(v) for v in value):
        return {
            "type": "tensor_list",
            "length": len(value),
            "items": [tensor_stats(v) for v in value],
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "value": list(value)}
    return {"type": type(value).__name__, "value": value}


def save_artifact(
    sample_dir: Path,
    manifest: dict[str, Any],
    name: str,
    value: Any,
    *,
    metadata_only: bool,
    skip_payload: bool = False,
    note: str | None = None,
) -> None:
    artifacts = manifest.setdefault("artifacts", {})
    info = describe_value(value)
    if note is not None:
        info["note"] = note

    if not metadata_only and not skip_payload:
        tensor_path = sample_dir / f"{name}.pt"
        torch.save(value, tensor_path)
        info["file"] = tensor_path.name
    else:
        info["file"] = None
        info["payload_skipped"] = True

    artifacts[name] = info


def as_long_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(torch.long)
    return torch.tensor(value, dtype=torch.long)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_inputs(processor: Any, prompt: str, image: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
    image_inputs = processor.image_processor(images=image, return_tensors="pt")
    inputs = processor._convert_images_audios_text_to_inputs(  # noqa: SLF001 - official remote-code helper
        image_inputs,
        {},
        prompt,
        return_tensors="pt",
    )
    inputs["input_mode"] = torch.tensor([IMAGE_MODE_VISION], dtype=torch.long)

    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "input_image_embeds": inputs["input_image_embeds"],
        "image_sizes": inputs["image_sizes"],
        "image_attention_mask": inputs["image_attention_mask"],
        "input_mode": inputs["input_mode"],
    }
    return model_inputs, image_inputs


def move_tensors(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


class Phi4MMVisionCapture(AbstractContextManager["Phi4MMVisionCapture"]):
    def __init__(self, image_embed: torch.nn.Module):
        self.image_embed = image_embed
        self.records: dict[str, list[torch.Tensor]] = {
            "hidden_states_minus2": [],
            "pooled_16x16_crop_features": [],
            "assembled_pre_projection_image_sequence": [],
            "final_projected_image_embeddings": [],
        }
        self._handles: list[Any] = []
        self._original_get_img_features: Any = None

    def __enter__(self) -> "Phi4MMVisionCapture":
        self._original_get_img_features = self.image_embed.get_img_features

        def wrapped_get_img_features(img_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            features = self._original_get_img_features(img_embeds, attention_mask=attention_mask)
            self._capture("pooled_16x16_crop_features", features)
            return features

        self.image_embed.get_img_features = wrapped_get_img_features

        def img_processor_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None and isinstance(output, (tuple, list)) and len(output) >= 3:
                hidden_states = output[2]
            if hidden_states is None:
                return
            self._capture("hidden_states_minus2", hidden_states[-2])

        def projection_pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs:
                self._capture("assembled_pre_projection_image_sequence", inputs[0])

        def projection_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self._capture("final_projected_image_embeddings", output)

        self._handles.append(self.image_embed.img_processor.register_forward_hook(img_processor_hook))
        self._handles.append(self.image_embed.img_projection.register_forward_pre_hook(projection_pre_hook))
        self._handles.append(self.image_embed.img_projection.register_forward_hook(projection_hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self._handles:
            handle.remove()
        if self._original_get_img_features is not None:
            self.image_embed.get_img_features = self._original_get_img_features
        return False

    def _capture(self, name: str, tensor: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        self.records[name].append(tensor.detach().cpu().clone())

    def single_or_list(self, name: str) -> torch.Tensor | list[torch.Tensor] | None:
        values = self.records[name]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return values


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def load_processor_and_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any | None]:
    local_files_only = not args.allow_downloads
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        use_fast=False,
    )
    if args.preprocess_only:
        return processor, None

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": parse_dtype(args.dtype),
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["_attn_implementation"] = args.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Failed to import an official Phi-4-MM remote-code dependency: "
            f"{missing!r}. Install the missing package, or use --preprocess-only "
            "for processor-only dumps."
        ) from exc
    model.eval()
    model.to(device)
    return processor, model


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def dump_one_image(
    *,
    label: str,
    image_path: Path,
    image_kind: str,
    prompt: str,
    processor: Any,
    model: Any | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    sample_dir = args.output_dir / label
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    model_inputs_cpu, image_inputs = prepare_inputs(processor, prompt, image)
    image.close()

    manifest: dict[str, Any] = {
        "label": label,
        "image_path": str(image_path),
        "image_kind": image_kind,
        "prompt_file": "prompt.txt",
        "prompt": prompt,
        "created_at_utc": utc_now(),
        "preprocess_only": bool(args.preprocess_only),
        "notes": [],
        "artifacts": {},
    }
    write_text(sample_dir / "prompt.txt", prompt)

    save_artifact(
        sample_dir,
        manifest,
        "input_image_embeds",
        image_inputs["input_image_embeds"].detach().cpu(),
        metadata_only=args.metadata_only,
    )
    save_artifact(
        sample_dir,
        manifest,
        "image_sizes",
        image_inputs["image_sizes"].detach().cpu(),
        metadata_only=args.metadata_only,
    )
    save_artifact(
        sample_dir,
        manifest,
        "image_attention_mask",
        image_inputs["image_attention_mask"].detach().cpu(),
        metadata_only=args.metadata_only,
    )
    save_artifact(
        sample_dir,
        manifest,
        "num_img_tokens",
        as_long_tensor(image_inputs["num_img_tokens"]),
        metadata_only=args.metadata_only,
    )
    save_artifact(
        sample_dir,
        manifest,
        "input_ids",
        model_inputs_cpu["input_ids"].detach().cpu(),
        metadata_only=args.metadata_only,
    )
    save_artifact(
        sample_dir,
        manifest,
        "attention_mask",
        model_inputs_cpu["attention_mask"].detach().cpu(),
        metadata_only=args.metadata_only,
    )
    save_artifact(
        sample_dir,
        manifest,
        "input_mode",
        model_inputs_cpu["input_mode"].detach().cpu(),
        metadata_only=args.metadata_only,
        note="InputMode.VISION (1) for the official Phi-4-MM remote code.",
    )

    if model is None:
        manifest["notes"].append("Model forward was skipped because --preprocess-only was set.")
        (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    model_inputs = move_tensors(model_inputs_cpu, device)
    image_embed = model.model.embed_tokens_extend.image_embed

    with torch.inference_mode(), Phi4MMVisionCapture(image_embed) as capture:
        outputs = model(
            **model_inputs,
            use_cache=False,
            output_hidden_states=args.dump_decoder_hidden_minus2,
            return_dict=True,
            num_logits_to_keep=args.num_logits_to_keep,
        )

    first_token_logits = outputs.logits[:, -1, :].squeeze(0).detach().cpu()
    save_artifact(
        sample_dir,
        manifest,
        "first_token_logits",
        first_token_logits,
        metadata_only=args.metadata_only,
        note="Next-token logits at the end of the prompt; this is the first generated token distribution.",
    )

    for name in (
        "hidden_states_minus2",
        "pooled_16x16_crop_features",
        "assembled_pre_projection_image_sequence",
        "final_projected_image_embeddings",
    ):
        value = capture.single_or_list(name)
        if value is None:
            manifest["notes"].append(f"Capture missing: {name}")
            continue
        save_artifact(
            sample_dir,
            manifest,
            name,
            value,
            metadata_only=args.metadata_only,
            skip_payload=(name == "hidden_states_minus2" and args.skip_vision_hidden_states),
            note=(
                "SigLIP image processor hidden_states[-2], before Phi-4 image_token_compression."
                if name == "hidden_states_minus2"
                else None
            ),
        )

    if args.dump_decoder_hidden_minus2:
        decoder_hidden = outputs.hidden_states[-2].detach().cpu()
        save_artifact(
            sample_dir,
            manifest,
            "decoder_hidden_states_minus2",
            decoder_hidden,
            metadata_only=args.metadata_only,
            note="Decoder outputs.hidden_states[-2]; optional and not the SigLIP hidden state used by the image tower.",
        )

    if args.max_new_tokens > 0:
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }
        generation_config_path = args.model_path / "generation_config.json"
        if GenerationConfig is not None and generation_config_path.is_file():
            generation_kwargs["generation_config"] = GenerationConfig.from_pretrained(
                args.model_path,
                local_files_only=not args.allow_downloads,
            )
        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)
        new_tokens = generated_ids[:, model_inputs["input_ids"].shape[1] :].detach().cpu()
        save_artifact(
            sample_dir,
            manifest,
            "generated_ids",
            new_tokens,
            metadata_only=args.metadata_only,
        )
        generated_text = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        write_text(sample_dir / "generated.txt", generated_text)
        manifest["generated_text_file"] = "generated.txt"

    (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.prompt = validate_prompt(args.prompt)
    images = resolve_images(args)

    configure_determinism(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor/model from {args.model_path}", file=sys.stderr)
    print(f"Device={device}, dtype={args.dtype}, attn={args.attn_implementation}", file=sys.stderr)
    versions = package_versions()
    print(
        "HF deps: "
        + ", ".join(f"{name}={version}" for name, version in versions.items()),
        file=sys.stderr,
    )
    processor, model = load_processor_and_model(args, device)

    root_manifest: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "dtype_arg": args.dtype,
        "attn_implementation_arg": args.attn_implementation,
        "local_files_only": not args.allow_downloads,
        "seed": args.seed,
        "package_versions": versions,
        "prompt": args.prompt,
        "metadata_only": bool(args.metadata_only),
        "preprocess_only": bool(args.preprocess_only),
        "samples": [],
        "notes": [],
    }
    if args.simple_object_image is None:
        root_manifest["notes"].append(
            "No --simple-object-image was supplied. The harness supports it, but only the selected images were dumped."
        )

    for label, image_path, image_kind in images:
        print(f"Dumping {label}: {image_path}", file=sys.stderr)
        sample_manifest = dump_one_image(
            label=label,
            image_path=image_path,
            image_kind=image_kind,
            prompt=args.prompt,
            processor=processor,
            model=model,
            args=args,
            device=device,
        )
        root_manifest["samples"].append(
            {
                "label": label,
                "image_path": str(image_path),
                "manifest": f"{label}/manifest.json",
                "artifact_names": sorted(sample_manifest.get("artifacts", {}).keys()),
            }
        )

    (args.output_dir / "manifest.json").write_text(json.dumps(root_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {args.output_dir / 'manifest.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
