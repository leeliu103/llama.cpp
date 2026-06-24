from __future__ import annotations

import json
import math

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, SentencePieceTokenTypes, TextModel, gguf, logger


@ModelBase.register("PhiForCausalLM")
class Phi2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.PHI2

    def set_gguf_parameters(self):
        rot_pct = self.rope_parameters["partial_rotary_factor"]
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])

        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings"]))

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(4 * n_embd)
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_epsilon", "layer_norm_eps"]))
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_add_bos_token(False)


@ModelBase.register("Phi3ForCausalLM", "Phi4ForCausalLMV")
class Phi3MiniModel(TextModel):
    model_arch = gguf.MODEL_ARCH.PHI3

    def set_vocab(self):
        # Phi-4 model uses GPT2Tokenizer
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                tokenizer_class = tokenizer_config_json['tokenizer_class']
                if tokenizer_class == 'GPT2Tokenizer':
                    return self._set_vocab_gpt2()

        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise ValueError(f'Error: Missing {tokenizer_path}')

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, foken_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token = foken_data["content"].encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        tokenizer_file = self.dir_model / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
                added_tokens = tokenizer_json.get("added_tokens", [])
                for foken_data in added_tokens:
                    token_id = int(foken_data["id"])
                    token = foken_data["content"].encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        n_head_kv = self.find_hparam(["num_key_value_heads", "n_head_kv"])
        rms_eps = self.find_hparam(["rms_norm_eps"])
        max_pos_embds = self.find_hparam(["n_positions", "max_position_embeddings"])
        orig_max_pos_embds = self.rope_parameters["original_max_position_embeddings"]
        rot_pct = self.rope_parameters.get("partial_rotary_factor", 1.0)
        rope_dims = int(rot_pct * n_embd) // n_head

        self.gguf_writer.add_context_length(max_pos_embds)
        self.gguf_writer.add_rope_scaling_orig_ctx_len(orig_max_pos_embds)
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        self.gguf_writer.add_rope_dimension_count(rope_dims)
        self.gguf_writer.add_rope_freq_base(self.rope_parameters.get("full_attention", self.rope_parameters)["rope_theta"])
        self.gguf_writer.add_file_type(self.ftype)
        sliding_window = self.hparams.get("sliding_window")
        # use zero value of sliding_window to distinguish Phi-4 from other PHI3 models
        if sliding_window is None:
            sliding_window = 0
        self.gguf_writer.add_sliding_window(sliding_window)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        max_pos_embds = self.find_hparam(["n_positions", "max_position_embeddings"])
        orig_max_pos_embds = self.rope_parameters["original_max_position_embeddings"]
        rot_pct = self.rope_parameters.get("partial_rotary_factor", 1.0)
        rope_dims = int(rot_pct * n_embd) // n_head

        # write rope scaling for long context (128k) model
        long_factors = self.rope_parameters.get('long_factor')
        short_factors = self.rope_parameters.get('short_factor')
        if not long_factors:
            return

        scale = max_pos_embds / orig_max_pos_embds

        rope_scaling_type = self.rope_parameters.get('rope_type', '').lower()
        if len(rope_scaling_type) == 0:
            raise KeyError('Missing the required key rope_scaling.type')

        if rope_scaling_type == 'su' or rope_scaling_type == 'longrope':
            attn_factor = math.sqrt(1 + math.log(scale) / math.log(orig_max_pos_embds)) if scale > 1.0 else 1.0
        elif rope_scaling_type == 'yarn':
            attn_factor = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0
        else:
            raise NotImplementedError(f'The rope scaling type {rope_scaling_type} is not supported yet')

        self.gguf_writer.add_rope_scaling_attn_factors(attn_factor)

        if long_factors is None or short_factors is None:
            raise KeyError('Missing the required key rope_scaling.long_factor or rope_scaling_short_factor')

        if len(long_factors) != len(short_factors) or len(long_factors) != rope_dims / 2:
            raise ValueError(f'The length of rope long and short factors must be {rope_dims / 2}. long_factors = {len(long_factors)}, short_factors = {len(short_factors)}.')

        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_LONG), torch.tensor(long_factors, dtype=torch.float32))
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_SHORT), torch.tensor(short_factors, dtype=torch.float32))


@ModelBase.register("Phi4MMForCausalLM")
class Phi4MMTextModel(Phi3MiniModel):
    model_arch = gguf.MODEL_ARCH.PHI3

    def __init__(self, *args, **kwargs):
        self._vision_lora_scale: float | None = None
        super().__init__(*args, **kwargs)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_vision_lora_scale(self) -> float:
        if self._vision_lora_scale is not None:
            return self._vision_lora_scale

        vision_lora = self.hparams.get("vision_lora") or {}
        lora_alpha = vision_lora.get("lora_alpha")
        rank = vision_lora.get("r")
        if lora_alpha is None or rank is None:
            logger.warning("Phi4MM vision LoRA config is missing lora_alpha/r; falling back to scale 2.0")
            self._vision_lora_scale = 2.0
        else:
            self._vision_lora_scale = float(lora_alpha) / float(rank)

        logger.info(f"Phi4MM vision LoRA merge scale = {self._vision_lora_scale}")
        return self._vision_lora_scale

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item

        if name.startswith("model.embed_tokens_extend."):
            return None

        if ".lora_A.speech." in name or ".lora_B.speech." in name:
            return None

        if ".lora_A.vision." in name or ".lora_B.vision." in name:
            return ModelBase.filter_tensors(item)

        return super().filter_tensors(item)

    def merge_vision_lora(self, base: Tensor, name: str) -> Tensor:
        prefix = name.removesuffix(".base_layer.weight")
        lora_a_name = f"{prefix}.lora_A.vision.weight"
        lora_b_name = f"{prefix}.lora_B.vision.weight"

        if lora_a_name not in self.model_tensors or lora_b_name not in self.model_tensors:
            raise KeyError(f"Missing Phi4MM vision LoRA tensors for {name!r}")

        lora_a = self.model_tensors[lora_a_name]().float()
        lora_b = self.model_tensors[lora_b_name]().float()

        if lora_a.ndim != 2 or lora_b.ndim != 2 or base.ndim != 2:
            raise ValueError(f"Unexpected Phi4MM LoRA tensor rank for {name!r}")
        if lora_b.shape[1] != lora_a.shape[0] or lora_b.shape[0] != base.shape[0] or lora_a.shape[1] != base.shape[1]:
            raise ValueError(
                f"Phi4MM LoRA shape mismatch for {name!r}: "
                f"base={tuple(base.shape)}, A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )

        return base.float() + self.get_vision_lora_scale() * (lora_b @ lora_a)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".base_layer.weight"):
            merged_name = name.removesuffix(".base_layer.weight") + ".weight"
            data_torch = self.merge_vision_lora(data_torch, name)
            yield from super().modify_tensors(data_torch, merged_name, bid)
            return

        if ".lora_A." in name or ".lora_B." in name:
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Phi4MMForCausalLM")
class Phi4MMVisionMmprojModel(MmprojModel):
    _IMAGE_PREFIX = "model.embed_tokens_extend.image_embed."
    _IMG_PROCESSOR_PREFIX = _IMAGE_PREFIX + "img_processor."
    _IMG_PROJECTION_PREFIX = _IMAGE_PREFIX + "img_projection."
    _LOCAL_IMG_PROCESSOR_PREFIX = "model.vision_model."
    _LOCAL_IMG_PROJECTION_PREFIX = "phi4mm.img_projection."
    _LOCAL_GLB_GN = "phi4mm.glb_GN"
    _LOCAL_SUB_GN = "phi4mm.sub_GN"

    _IMAGE_SIZE = 448
    _PATCH_SIZE = 14
    _HIDDEN_SIZE = 1152
    _INTERMEDIATE_SIZE = 4304
    _HEAD_COUNT = 16
    _LAYER_NORM_EPS = 1e-6
    _EXPORT_LAYERS = 26
    _DYNAMIC_HD = 36

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        self.dynamic_hd = int(self.preprocessor_config.get("dynamic_hd", self._DYNAMIC_HD))
        if self.dynamic_hd != self._DYNAMIC_HD:
            raise ValueError(f"Phi4MM mmproj converter expects dynamic_hd={self._DYNAMIC_HD}, got {self.dynamic_hd}")

        self.preprocessor_config.setdefault("image_mean", [0.5, 0.5, 0.5])
        self.preprocessor_config.setdefault("image_std", [0.5, 0.5, 0.5])

        self.block_count = self._EXPORT_LAYERS
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)

    def get_vision_config(self):
        vision_config = super().get_vision_config()
        if vision_config is not None:
            return vision_config

        image_config = self.global_config.get("embd_layer", {}).get("image_embd_layer", {})
        expected_image_config = {
            "crop_size": self._IMAGE_SIZE,
            "image_token_compression_cls": "avg_pool_2d",
            "projection_cls": "mlp",
            "use_hd_transform": True,
            "with_learnable_separator": True,
            "hd_transform_order": "sub_glb",
        }
        for key, expected in expected_image_config.items():
            value = image_config.get(key)
            if value is not None and value != expected:
                raise ValueError(f"Phi4MM mmproj converter expects image_embd_layer.{key}={expected!r}, got {value!r}")

        return {
            "architectures": ["Phi4MMVisionMmprojModel"],
            "image_size": self._IMAGE_SIZE,
            "patch_size": self._PATCH_SIZE,
            "hidden_size": self._HIDDEN_SIZE,
            "intermediate_size": self._INTERMEDIATE_SIZE,
            "num_attention_heads": self._HEAD_COUNT,
            "num_hidden_layers": self._EXPORT_LAYERS,
            "layer_norm_eps": self._LAYER_NORM_EPS,
        }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PHI4MM)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(self._LAYER_NORM_EPS)
        self.gguf_writer.add_vision_preproc_max_tiles(self.dynamic_hd)

    @classmethod
    def _is_dropped_img_processor_tensor(cls, local_name: str) -> bool:
        if local_name.startswith(("post_layernorm.", "head.")):
            return True

        layer_prefix = "encoder.layers."
        if not local_name.startswith(layer_prefix):
            return False

        rest = local_name[len(layer_prefix):]
        layer_idx, has_sep, _ = rest.partition(".")
        return has_sep and layer_idx.isdecimal() and int(layer_idx) >= cls._EXPORT_LAYERS

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(cls._IMG_PROCESSOR_PREFIX):
            local_name = name.removeprefix(cls._IMG_PROCESSOR_PREFIX)
            if cls._is_dropped_img_processor_tensor(local_name):
                return None
            return super().filter_tensors((cls._LOCAL_IMG_PROCESSOR_PREFIX + local_name, gen))

        if name.startswith(cls._IMG_PROJECTION_PREFIX):
            local_name = name.removeprefix(cls._IMG_PROJECTION_PREFIX)
            layer_idx, has_sep, _ = local_name.partition(".")
            if not has_sep or layer_idx not in {"0", "2"}:
                return None
            return super().filter_tensors((cls._LOCAL_IMG_PROJECTION_PREFIX + local_name, gen))

        if name == cls._IMAGE_PREFIX + "glb_GN":
            return super().filter_tensors((cls._LOCAL_GLB_GN, gen))

        if name == cls._IMAGE_PREFIX + "sub_GN":
            return super().filter_tensors((cls._LOCAL_SUB_GN, gen))

        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith(self._LOCAL_IMG_PROCESSOR_PREFIX):
            yield from super().modify_tensors(data_torch, name, bid)
            return

        if name.startswith(self._LOCAL_IMG_PROJECTION_PREFIX):
            local_name = name.removeprefix(self._LOCAL_IMG_PROJECTION_PREFIX)
            layer_idx = int(local_name.split(".", maxsplit=1)[0])
            suffix = ".bias" if local_name.endswith(".bias") else ".weight"
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, layer_idx, suffix=suffix), data_torch)
            return

        if name == self._LOCAL_GLB_GN:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_ENC_EMBD_VSEP, suffix=""), data_torch)
            return

        if name == self._LOCAL_SUB_GN:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_ENC_EMBD_IMGNL, suffix=""), data_torch)
            return

        return


@ModelBase.register("Phi4ForCausalLMV")
class Phi4VisionMmprojModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        self.vision_total_layers = int(self.find_vparam(self.n_block_keys))
        if self.vision_total_layers < 2:
            raise ValueError(
                f"Phi-4 vision mmproj conversion requires at least 2 vision layers, got {self.vision_total_layers}"
            )

        # Phi-4 uses SigLIP2 hidden_states[-2], so export one fewer encoder block and
        # drop post-layernorm/head weights. This makes the GGUF runtime output match
        # the feature map consumed by the patched siglip.cpp Phi-4 projector path.
        self.vision_export_layers = self.vision_total_layers - 1
        self.vision_last_layer_idx = self.vision_total_layers - 1

        for key in self.n_block_keys:
            if key in self.hparams_vision:
                self.hparams_vision[key] = self.vision_export_layers
                break

        self.block_count = self.vision_export_layers
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)

        patch_size = self.preprocessor_config.get("patch_size")
        if patch_size is None:
            raise KeyError("Phi-4 vision mmproj conversion requires patch_size in preprocessor_config.json")

        self.hparams_vision["patch_size"] = patch_size

        pos_emb_name = next(
            (
                name for name in self.model_tensors
                if name.endswith("vision_model.embeddings.position_embedding.weight")
            ),
            None,
        )
        if pos_emb_name is None:
            raise KeyError("Phi-4 vision mmproj conversion could not find position_embedding.weight")

        pos_emb_shape = self.model_tensors[pos_emb_name]().shape
        base_grid_tokens = int(pos_emb_shape[0])
        grid_side = math.isqrt(base_grid_tokens)
        if grid_side * grid_side != base_grid_tokens:
            raise ValueError(f"Unexpected Phi-4 position embedding shape: {tuple(pos_emb_shape)}")

        self.hparams_vision["image_size"] = grid_side * patch_size

        min_num_patches = self.preprocessor_config.get("min_num_patches", self.global_config.get("min_num_patches"))
        max_num_patches = self.preprocessor_config.get("max_num_patches", self.global_config.get("max_num_patches"))
        if min_num_patches is None or max_num_patches is None:
            raise KeyError("Phi-4 vision mmproj conversion requires min_num_patches and max_num_patches")

        self.min_pixels = int(min_num_patches) * patch_size * patch_size
        self.max_pixels = int(max_num_patches) * patch_size * patch_size

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PHI4)
        self.gguf_writer.add_vision_min_pixels(self.min_pixels)
        self.gguf_writer.add_vision_max_pixels(self.max_pixels)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-6))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        name = name.replace("model.vision_tower.vision_tower.", "vision_tower.")

        if not name.startswith(("vision_tower.", "model.mm_projector.", "mm_projector.")):
            return None

        if ".vision_model.head." in name:
            return None

        if ".vision_model.post_layernorm." in name:
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("vision_tower."):
            if bid is not None and bid == self.vision_last_layer_idx:
                return

            if name.endswith("vision_model.embeddings.patch_embedding.weight"):
                assert self.hparams_vision is not None
                if data_torch.ndim != 2:
                    raise ValueError(f"Unexpected Phi-4 patch embedding shape: {tuple(data_torch.shape)}")

                patch_area = self.hparams_vision["patch_size"] ** 2
                in_features = data_torch.shape[1]
                if in_features % patch_area != 0:
                    raise ValueError(
                        f"Phi-4 patch embedding input dim {in_features} is not divisible by patch area {patch_area}"
                    )

                num_channels = in_features // patch_area
                patch_size = self.hparams_vision["patch_size"]
                data_torch = data_torch.view(data_torch.shape[0], patch_size, patch_size, num_channels)
                data_torch = data_torch.permute(0, 3, 1, 2)

            yield from super().modify_tensors(data_torch, name, bid)
            return

        if name.startswith(("model.mm_projector.", "mm_projector.")):
            local_name = name
            local_name = local_name.replace("model.mm_projector.", "")
            local_name = local_name.replace("mm_projector.", "")

            if not (local_name.startswith("0.") or local_name.startswith("2.")):
                return

            suffix = ".bias" if local_name.endswith(".bias") else ".weight"
            mm_idx = int(local_name.split(".", maxsplit=1)[0])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, mm_idx, suffix=suffix), data_torch)
            return

        return


@ModelBase.register("PhiMoEForCausalLM")
class PhiMoeModel(Phi3MiniModel):
    model_arch = gguf.MODEL_ARCH.PHIMOE

    _experts: list[dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_used_count(self.find_hparam(["num_experts_per_tok", "num_experts_per_token"]))
        self.gguf_writer.add_expert_count(self.find_hparam(["num_local_experts", "num_experts"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
