from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.models.action_dit import ActionDiT
from fastwam.models.helpers.io import ModelConfig, load_state_dict
from fastwam.models.mot import MoT
from .utils import (
    as_hw,
    as_optional_path,
    as_plain_dict,
    build_world_action_mot_mask,
    compute_action_flow_loss,
    parse_mot_action_to_world_config,
)
from fastwam.models.representation_encoders import build_representation_encoder
from fastwam.models.representation_codecs import (
    CausalRunningChannelNorm,
    CausalCodecFeatureDecoder,
    FAEAttentionCausalCodec,
    FrozenRandomCausalCodec,
    LearnableRandomCausalCodec,
)
from fastwam.models.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _parse_temporal_indices(value: Any, name: str, min_steps: int) -> Optional[list[int]]:
    if value in (None, "", "null"):
        return None
    indices = [int(v) for v in value]
    if len(indices) < min_steps:
        raise ValueError(f"`{name}` must contain at least {min_steps} frames.")
    if indices[0] != 0:
        raise ValueError(f"`{name}` must start with 0 for first-frame conditioning.")
    if min(indices) < 0:
        raise ValueError(f"`{name}` must be non-negative, got {indices}.")
    if sorted(set(indices)) != indices:
        raise ValueError(f"`{name}` must be strictly increasing unique indices, got {indices}.")
    return indices


def _parse_temporal_groups(value: Any, name: str, min_steps: int) -> Optional[list[list[int]]]:
    if value in (None, "", "null"):
        return None
    groups = [[int(idx) for idx in group] for group in value]
    if len(groups) < min_steps:
        raise ValueError(f"`{name}` must contain at least {min_steps} groups.")
    for group in groups:
        if not group:
            raise ValueError(f"`{name}` cannot contain empty groups: {groups}.")
        if min(group) < 0:
            raise ValueError(f"`{name}` must be non-negative, got {groups}.")
    if groups[0][0] != 0:
        raise ValueError(f"`{name}` must start with frame 0 for first-frame conditioning.")
    return groups


def _validate_causal_codec_temporal_groups(
    groups: Optional[list[list[int]]],
    *,
    temporal_padding: str = "left_zero",
) -> None:
    if not groups:
        raise ValueError("Causal representation codec requires `representation.temporal_groups`.")
    temporal_padding = str(temporal_padding).lower()
    if temporal_padding == "left_zero":
        if groups[0] != [0]:
            raise ValueError(
                "Left-zero causal representation codec requires the first temporal group "
                f"to be exactly [0], got {groups[0]}."
            )
        paired_groups = groups[1:]
    elif temporal_padding == "none":
        paired_groups = groups
    else:
        raise ValueError(f"Unsupported codec temporal padding: {temporal_padding!r}.")

    invalid_groups = [group for group in paired_groups if len(group) != 2]
    if invalid_groups:
        raise ValueError(
            "Causal representation codec requires every paired temporal group to contain exactly "
            f"two frames, got invalid groups {invalid_groups} from {groups}."
        )

    flat_indices = [idx for group in groups for idx in group]
    if sorted(set(flat_indices)) != flat_indices:
        raise ValueError(
            "Causal representation codec requires temporal group indices to be strictly increasing "
            f"and unique across groups, got {groups}."
        )


@dataclass(frozen=True)
class RepresentationConfig:
    state_space: str
    noise_timestep_mode: str
    target_dim: int
    latent_spatial_size: tuple[int, int]
    normalize_target_mode: str
    normalize_target: bool
    latent_stats_path: Optional[str]
    latent_stats_eps: float
    temporal_align_mode: str
    temporal_groups: Optional[list[list[int]]]
    temporal_indices: Optional[list[int]]

    @classmethod
    def from_dict(
        cls,
        cfg: Optional[dict[str, Any]],
        *,
        default_target_dim: int,
        default_latent_spatial_size: tuple[int, int] = (12, 10),
    ) -> "RepresentationConfig":
        cfg = as_plain_dict(cfg)
        encoder_cfg = as_plain_dict(cfg.get("encoder", {}))

        state_space = str(cfg.get("state_space", "absolute")).lower()
        if state_space not in {"absolute", "delta"}:
            raise ValueError(
                "representation.state_space must be one of {'absolute', 'delta'}, "
                f"got {state_space!r}."
            )

        noise_timestep_mode = str(cfg.get("noise_timestep_mode", "shared")).lower()
        if noise_timestep_mode not in {"shared", "per_frame"}:
            raise ValueError(
                "representation.noise_timestep_mode must be one of {'shared', 'per_frame'}, "
                f"got {noise_timestep_mode!r}."
            )

        target_dim = int(
            cfg.get(
                "target_dim",
                encoder_cfg.get("output_dim", default_target_dim),
            )
        )
        latent_spatial_size = as_hw(
            cfg.get("latent_spatial_size", None),
            default=default_latent_spatial_size,
        )

        normalize_target = cfg.get("normalize_target", False)
        if isinstance(normalize_target, str):
            normalize_target_mode = normalize_target.lower()
        else:
            normalize_target_mode = "per_sample" if bool(normalize_target) else "none"
        if normalize_target_mode in {"false", "off", "no", "0"}:
            normalize_target_mode = "none"
        if normalize_target_mode in {"true", "on", "yes", "1"}:
            normalize_target_mode = "per_sample"
        if normalize_target_mode not in {"none", "per_sample", "dataset"}:
            raise ValueError(
                "RA representation.normalize_target must be false/true or one of "
                "{'none', 'per_sample', 'dataset'}, "
                f"got {normalize_target!r}."
            )

        temporal_align_mode = str(cfg.get("temporal_align_mode", "strict")).lower()
        if temporal_align_mode not in {"strict", "interpolate"}:
            raise ValueError(f"Unsupported RA representation.temporal_align_mode: {temporal_align_mode}")

        temporal_groups = _parse_temporal_groups(
            cfg.get("temporal_groups", None),
            "RA representation.temporal_groups",
            min_steps=2,
        )
        temporal_indices = _parse_temporal_indices(
            cfg.get("temporal_indices", None if temporal_groups is not None else [0, 4, 8]),
            "RA representation.temporal_indices",
            min_steps=2,
        )
        if temporal_groups is not None and temporal_indices is not None:
            raise ValueError("Use only one of RA `representation.temporal_indices` and `representation.temporal_groups`.")

        return cls(
            state_space=state_space,
            noise_timestep_mode=noise_timestep_mode,
            target_dim=target_dim,
            latent_spatial_size=latent_spatial_size,
            normalize_target_mode=normalize_target_mode,
            normalize_target=normalize_target_mode != "none",
            latent_stats_path=as_optional_path(
                cfg.get("latent_stats_path", cfg.get("normalization_stat_path", None))
            ),
            latent_stats_eps=float(cfg.get("latent_stats_eps", 1e-5)),
            temporal_align_mode=temporal_align_mode,
            temporal_groups=temporal_groups,
            temporal_indices=temporal_indices,
        )

    @property
    def temporal_steps(self) -> int:
        if self.temporal_groups is not None:
            return len(self.temporal_groups)
        if self.temporal_indices is not None:
            return len(self.temporal_indices)
        return 3

    def token_count_for_shift(self, *, shift_scope: str) -> tuple[int, int]:
        scope = str(shift_scope).lower()
        if scope == "future_tokens":
            time_steps = max(self.temporal_steps - 1, 1)
        elif scope == "all_tokens":
            time_steps = max(self.temporal_steps, 1)
        else:
            raise ValueError(f"Unsupported RA representation_scheduler.shift_scope: {shift_scope!r}.")
        height, width = self.latent_spatial_size
        return self.target_dim * time_steps * height * width, time_steps

    def resolve_shift(
        self,
        *,
        value: Any,
        shift_base_dim: int,
        shift_scope: str,
        name: str,
    ) -> float:
        if not (isinstance(value, str) and value.strip().lower() == "auto"):
            return float(value)
        if shift_base_dim <= 0:
            raise ValueError(f"RA representation_scheduler.shift_base_dim must be positive, got {shift_base_dim}.")
        latent_dim, time_steps = self.token_count_for_shift(shift_scope=shift_scope)
        shift = math.sqrt(float(latent_dim) / float(shift_base_dim))
        height, width = self.latent_spatial_size
        logger.info(
            "Resolved RA %s=auto to %.4f from latent_dim=%d "
            "(C=%d T=%d H=%d W=%d base=%d scope=%s).",
            name,
            shift,
            latent_dim,
            self.target_dim,
            time_steps,
            height,
            width,
            shift_base_dim,
            str(shift_scope).lower(),
        )
        return float(shift)


def _resolve_wan_dit_pretrain_path(model_id: str) -> str | list[str]:
    config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    config.download_if_necessary()
    if config.path is None:
        raise ValueError(f"Could not resolve WAN DiT checkpoint for model_id={model_id}.")
    return config.path


def _partial_load_shape_compatible(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    module_name: str,
) -> dict[str, Any]:
    current_state = module.state_dict()
    compatible = {}
    skipped_shape = []
    skipped_unexpected = []
    for key, value in state_dict.items():
        if key not in current_state:
            skipped_unexpected.append(key)
            continue
        target = current_state[key]
        if tuple(value.shape) != tuple(target.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(target.shape)))
            continue
        compatible[key] = value.to(device=target.device, dtype=target.dtype)

    load_result = module.load_state_dict(compatible, strict=False)
    logger.info(
        "Partial-loaded %s: loaded=%d skipped_shape=%d skipped_unexpected=%d missing_after_load=%d.",
        module_name,
        len(compatible),
        len(skipped_shape),
        len(skipped_unexpected),
        len(load_result.missing_keys),
    )
    if skipped_shape:
        logger.info(
            "%s shape-mismatched keys skipped: %s",
            module_name,
            skipped_shape[:20],
        )
    if skipped_unexpected:
        logger.info(
            "%s unexpected keys skipped: %s%s",
            module_name,
            skipped_unexpected[:20],
            "..." if len(skipped_unexpected) > 20 else "",
        )
    return {
        "loaded": len(compatible),
        "skipped_shape": skipped_shape,
        "skipped_unexpected": skipped_unexpected,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


class RA(nn.Module):
    """Representation-Diffusion + Action MoT.

    RA keeps FastWAM's MoT/action training pattern, but replaces the WAN/VAE
    video latent branch with an online frozen visual representation branch.
    """

    def __init__(
        self,
        representation_expert: WanVideoDiT,
        action_expert: ActionDiT,
        mot: MoT,
        representation_encoder: nn.Module,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        representation_train_shift: float = 5.0,
        representation_infer_shift: float = 5.0,
        representation_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_representation: float = 1.0,
        loss_lambda_action: float = 1.0,
        representation: Optional[dict[str, Any]] = None,
        mot_conditioning: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.representation_expert = representation_expert
        # Keep FastWAM-compatible names for shared trainer/MoT helpers.
        self.video_expert = self.representation_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.representation_encoder = representation_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self._capture_representation_viz = False
        self._last_representation_viz = None

        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_representation_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=representation_num_train_timesteps,
            shift=representation_train_shift,
        )
        self.infer_representation_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=representation_num_train_timesteps,
            shift=representation_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_representation_scheduler
        self.infer_scheduler = self.infer_representation_scheduler

        representation_dict = as_plain_dict(representation)
        self.representation_config = RepresentationConfig.from_dict(
            representation_dict,
            default_target_dim=int(
                getattr(
                    self.representation_encoder,
                    "output_dim",
                    getattr(self.representation_expert, "in_dim", 2048),
                )
            ),
        )
        self.representation_state_space = self.representation_config.state_space
        self.representation_noise_timestep_mode = self.representation_config.noise_timestep_mode
        self.target_dim = self.representation_config.target_dim
        self.encoder_output_dim = int(
            getattr(self.representation_encoder, "output_dim", self.target_dim)
        )
        codec_cfg = as_plain_dict(representation_dict.get("codec", None), default={})
        self.representation_codec_enabled = bool(codec_cfg.get("enabled", False))
        self.representation_codec = None
        self.representation_codec_trainable = False
        self.representation_codec_gradient_mode = "none"
        self.representation_codec_temporal_padding = "left_zero"
        self.representation_codec_uses_history = False
        self.codec_latent_norm = None
        self.codec_decoder = None
        self.codec_decoder_loss_weight = 0.0
        self.codec_decoder_trainable = False
        if self.representation_codec_enabled:
            codec_type = str(codec_cfg.get("type", "frozen_random")).lower()
            if codec_type not in {"frozen_random", "learnable_random", "fae_attention"}:
                raise ValueError(
                    "RARAE representation.codec.type must be one of "
                    "{'frozen_random', 'learnable_random', 'fae_attention'}, "
                    f"got {codec_type!r}."
                )
            self.representation_codec_trainable = codec_type in {
                "learnable_random",
                "fae_attention",
            }
            self.representation_codec_temporal_padding = str(
                codec_cfg.get("temporal_padding", "left_zero")
            ).lower()
            if self.representation_codec_temporal_padding not in {"left_zero", "none"}:
                raise ValueError(
                    "representation.codec.temporal_padding must be one of "
                    "{'left_zero', 'none'}, "
                    f"got {self.representation_codec_temporal_padding!r}."
                )
            self.representation_codec_uses_history = (
                self.representation_codec_temporal_padding == "none"
            )
            if self.representation_codec_trainable:
                self.representation_codec_gradient_mode = str(
                    codec_cfg.get("gradient_mode", "condition_and_action")
                ).lower()
                if self.representation_codec_gradient_mode not in {
                    "condition_and_action",
                    "action_only",
                }:
                    raise ValueError(
                        "Trainable representation codec gradient_mode must be one of "
                        "{'condition_and_action', 'action_only'}, "
                        f"got {self.representation_codec_gradient_mode!r}."
                    )
                if self.representation_state_space != "absolute":
                    raise ValueError(
                        "Learnable action-shaped codec requires representation.state_space='absolute'."
                    )
            codec_output_dim = int(codec_cfg.get("output_dim", self.target_dim))
            if codec_output_dim != self.target_dim:
                raise ValueError(
                    "representation.codec.output_dim must match representation.target_dim: "
                    f"codec={codec_output_dim}, target={self.target_dim}."
                )
            decoder_cfg = as_plain_dict(codec_cfg.get("decoder", None), default={})
            decoder_enabled = bool(decoder_cfg.get("enabled", False))
            if codec_type == "fae_attention":
                camera_layout = str(
                    getattr(self.representation_encoder, "camera_layout", "none")
                ).lower()
                num_cameras = int(
                    getattr(self.representation_encoder, "num_cameras", 1)
                )
                if camera_layout != "robotwin" or num_cameras != 3:
                    raise ValueError(
                        "FAE attention codec currently requires the three-camera RobotWin layout, "
                        f"got layout={camera_layout!r}, num_cameras={num_cameras}."
                    )
                self.representation_codec = FAEAttentionCausalCodec(
                    input_dim=self.encoder_output_dim,
                    output_dim=codec_output_dim,
                    qk_dim=int(codec_cfg.get("qk_dim", 1024)),
                    num_heads=int(codec_cfg.get("num_heads", 8)),
                    kernel_size=codec_cfg.get("kernel_size", (2, 2, 2)),
                    stride=codec_cfg.get("stride", (2, 2, 2)),
                    seed=int(codec_cfg.get("seed", 0)),
                    temporal_padding=self.representation_codec_temporal_padding,
                    norm_eps=float(codec_cfg.get("norm_eps", 1e-6)),
                    rope_base=float(codec_cfg.get("rope_base", 10000.0)),
                    position_scale=float(codec_cfg.get("position_scale", 16.0)),
                ).to(dtype=torch_dtype)
            else:
                codec_cls = (
                    LearnableRandomCausalCodec
                    if self.representation_codec_trainable
                    else FrozenRandomCausalCodec
                )
                self.representation_codec = codec_cls(
                    input_dim=self.encoder_output_dim,
                    output_dim=codec_output_dim,
                    kernel_size=codec_cfg.get("kernel_size", (2, 2, 2)),
                    stride=codec_cfg.get("stride", (2, 2, 2)),
                    seed=int(codec_cfg.get("seed", 0)),
                    temporal_padding=self.representation_codec_temporal_padding,
                ).to(dtype=torch_dtype)
            latent_norm_cfg = as_plain_dict(codec_cfg.get("latent_norm", None), default={})
            latent_norm_enabled = bool(latent_norm_cfg.get("enabled", True))
            latent_norm_type = str(latent_norm_cfg.get("type", "causal_running")).lower()
            if latent_norm_enabled:
                if latent_norm_type != "causal_running":
                    raise ValueError(
                        "representation.codec.latent_norm.type must be 'causal_running', "
                        f"got {latent_norm_type!r}."
                    )
                self.codec_latent_norm = CausalRunningChannelNorm(
                    codec_output_dim,
                    eps=float(latent_norm_cfg.get("eps", 1e-5)),
                    momentum=float(latent_norm_cfg.get("momentum", 0.1)),
                )
            if decoder_enabled:
                self.codec_decoder = CausalCodecFeatureDecoder(
                    input_dim=codec_output_dim,
                    output_dim=self.encoder_output_dim,
                    drop_first_temporal_output=(
                        self.representation_codec_temporal_padding == "left_zero"
                    ),
                ).to(dtype=torch_dtype)
                self.codec_decoder_loss_weight = float(decoder_cfg.get("loss_weight", 1.0))
                self.codec_decoder_trainable = bool(decoder_cfg.get("trainable", True))
                if self.codec_decoder_loss_weight < 0:
                    raise ValueError("representation.codec.decoder.loss_weight must be non-negative.")
            _validate_causal_codec_temporal_groups(
                self.representation_config.temporal_groups,
                temporal_padding=self.representation_codec_temporal_padding,
            )
            logger.info(
                "Enabled %s causal representation codec: input_dim=%d output_dim=%d "
                "output_spatial=%s temporal_groups=%s temporal_padding=%s gradient_mode=%s.",
                codec_type,
                self.encoder_output_dim,
                self.target_dim,
                self.representation_config.latent_spatial_size,
                self.representation_config.temporal_groups,
                self.representation_codec_temporal_padding,
                self.representation_codec_gradient_mode,
            )
            if self.codec_decoder is not None:
                logger.info(
                    "Enabled detached codec feature decoder: %d -> %d channels, "
                    "loss_weight=%.4f trainable=%s.",
                    codec_output_dim,
                    self.encoder_output_dim,
                    self.codec_decoder_loss_weight,
                    self.codec_decoder_trainable,
                )
        elif self.encoder_output_dim != self.target_dim:
            raise ValueError(
                "Representation encoder output dim must match target dim when codec is disabled: "
                f"encoder={self.encoder_output_dim}, target={self.target_dim}."
            )
        if int(getattr(self.representation_expert, "in_dim", self.target_dim)) != self.target_dim:
            raise ValueError(
                "Representation expert `in_dim` must match representation target dim: "
                f"expert={getattr(self.representation_expert, 'in_dim', None)} target={self.target_dim}."
            )
        self.latent_spatial_size = self.representation_config.latent_spatial_size
        self.normalize_target_mode = self.representation_config.normalize_target_mode
        self.normalize_target = self.representation_config.normalize_target
        self.latent_stats_path = self.representation_config.latent_stats_path
        self.register_buffer("latent_mean", None, persistent=False)
        self.register_buffer("latent_var", None, persistent=False)
        self.latent_stats_eps = self.representation_config.latent_stats_eps
        self.representation_decoder_path = as_optional_path(representation_dict.get("decoder_path"))
        self.representation_decoder = None
        if self.representation_codec_enabled and self.normalize_target_mode != "none":
            raise ValueError(
                "Codec-based RARAE normalizes the actual codec latent after compression; "
                "set representation.normalize_target='none' instead of applying RAEv2 "
                "feature statistics before the codec."
            )
        if self.normalize_target_mode == "dataset":
            if self.latent_stats_path is None:
                logger.warning(
                    "RA representation.normalize_target='dataset' was set without latent_stats_path; "
                    "falling back to no target normalization."
                )
                self.normalize_target_mode = "none"
                self.normalize_target = False
            else:
                stats = torch.load(self.latent_stats_path, map_location="cpu")
                mean = stats.get("mean", None)
                var = stats.get("var", stats.get("variance", None))
                if mean is None or var is None:
                    raise ValueError(f"Latent stats file must contain `mean` and `var`: {self.latent_stats_path}")
                self.latent_mean = mean.detach().float()
                self.latent_var = var.detach().float()
                logger.info("Loaded RA latent normalization stats from %s.", self.latent_stats_path)
        self.temporal_align_mode = self.representation_config.temporal_align_mode
        self.temporal_groups = self.representation_config.temporal_groups
        self.temporal_indices = self.representation_config.temporal_indices

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_representation = float(loss_lambda_representation)
        self.loss_lambda_action = float(loss_lambda_action)
        self.mot_action_to_world = parse_mot_action_to_world_config(mot_conditioning)
        self.mot_action_to_world_enabled = self.mot_action_to_world.enabled

        # Record the target dtype on the lazily loaded DINO teacher as well as
        # moving registered modules. Codec pretraining uses the same BF16 DINO
        # path, so leaving the WAM teacher at implicit FP32 would shift targets.
        self.to(device=self.device, dtype=self.torch_dtype)
        self._freeze_representation_encoder()

    @classmethod
    def from_config(
        cls,
        *,
        representation_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        representation_dit_pretrained_source: Optional[str] = None,
        representation_dit_pretrained_path: Optional[str] = None,
        representation_dit_pretrained_model_id: Optional[str] = None,
        mot_checkpoint_mixed_attn: bool = True,
        representation: Optional[dict[str, Any]] = None,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        representation_train_shift: float = 5.0,
        representation_infer_shift: float = 5.0,
        representation_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_representation: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_conditioning: Optional[dict[str, Any]] = None,
    ) -> "RA":
        representation = as_plain_dict(representation)
        representation_encoder = build_representation_encoder(representation)
        representation_expert = WanVideoDiT(**representation_dit_config).to(device=device, dtype=torch_dtype)
        representation_dit_pretrain_meta = None
        pretrain_source = None if representation_dit_pretrained_source in (None, "", "null") else str(representation_dit_pretrained_source).lower()
        if pretrain_source is not None:
            if pretrain_source not in {"wan", "path"}:
                raise ValueError(
                    "`representation_dit_pretrained_source` must be one of null, 'wan', or 'path', "
                    f"got {representation_dit_pretrained_source}."
                )
            if pretrain_source == "wan":
                if representation_dit_pretrained_path not in (None, "", "null"):
                    pretrain_path = representation_dit_pretrained_path
                else:
                    pretrain_path = _resolve_wan_dit_pretrain_path(
                        representation_dit_pretrained_model_id or "Wan-AI/Wan2.2-TI2V-5B"
                    )
            else:
                if representation_dit_pretrained_path in (None, "", "null"):
                    raise ValueError("`representation_dit_pretrained_path` is required when source='path'.")
                pretrain_path = representation_dit_pretrained_path
            state_dict = load_state_dict(pretrain_path, torch_dtype=torch_dtype, device="cpu")
            representation_dit_pretrain_meta = _partial_load_shape_compatible(
                representation_expert,
                state_dict,
                module_name="RA representation_expert",
            )
            representation_dit_pretrain_meta["path"] = pretrain_path
            representation_dit_pretrain_meta["source"] = pretrain_source
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(representation_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match representation expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(representation_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match representation expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(representation_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match representation expert.")
        mot = MoT(
            mixtures={"video": representation_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            representation_expert=representation_expert,
            action_expert=action_expert,
            mot=mot,
            representation_encoder=representation_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            representation_train_shift=representation_train_shift,
            representation_infer_shift=representation_infer_shift,
            representation_num_train_timesteps=representation_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_representation=loss_lambda_representation,
            loss_lambda_action=loss_lambda_action,
            representation=representation,
            mot_conditioning=mot_conditioning,
        )
        model.model_paths = {
            "representation_dit": (
                "RANDOM_INIT"
                if representation_dit_pretrain_meta is None
                else representation_dit_pretrain_meta["path"]
            ),
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        if representation_dit_pretrain_meta is not None:
            model.representation_dit_pretrain_meta = representation_dit_pretrain_meta
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        self.representation_encoder.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self

    def _normalize_representation_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.normalize_target_mode == "none":
            return features
        if self.normalize_target_mode == "per_sample":
            mean = features.mean(dim=(1, 2, 3, 4), keepdim=True)
            std = features.std(dim=(1, 2, 3, 4), keepdim=True)
            return (features - mean) / (std + 1e-6)
        if self.latent_mean is None or self.latent_var is None:
            raise ValueError("Dataset latent normalization requested but latent stats were not loaded.")
        mean = self.latent_mean.to(device=features.device, dtype=features.dtype)
        var = self.latent_var.to(device=features.device, dtype=features.dtype)
        if mean.ndim == 3:
            mean = mean.unsqueeze(0).unsqueeze(2)
        elif mean.ndim == 4:
            mean = mean.unsqueeze(2)
        elif mean.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported latent mean shape: {tuple(mean.shape)}")
        if var.ndim == 3:
            var = var.unsqueeze(0).unsqueeze(2)
        elif var.ndim == 4:
            var = var.unsqueeze(2)
        elif var.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported latent var shape: {tuple(var.shape)}")
        if mean.shape[-2:] != features.shape[-2:]:
            mean = F.interpolate(
                mean.flatten(0, 2).unsqueeze(0),
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).reshape(*mean.shape[:-2], features.shape[-2], features.shape[-1])
            var = F.interpolate(
                var.flatten(0, 2).unsqueeze(0),
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).reshape(*var.shape[:-2], features.shape[-2], features.shape[-1])
        return (features - mean) / torch.sqrt(var.clamp_min(0.0) + self.latent_stats_eps)

    def _freeze_representation_encoder(self):
        self.representation_encoder.eval()
        self.representation_encoder.requires_grad_(False)
        model = getattr(self.representation_encoder, "model", None)
        if model is not None:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    def set_representation_visualization_capture(self, enabled: bool):
        self._capture_representation_viz = bool(enabled)
        if not enabled:
            self._last_representation_viz = None

    def pop_last_representation_visualization(self):
        payload = self._last_representation_viz
        self._last_representation_viz = None
        return payload

    @staticmethod
    def _repr_latents_to_pca_rgb_strip(pred_repr: torch.Tensor, target_repr: torch.Tensor) -> torch.Tensor:
        if pred_repr.ndim != 5 or target_repr.ndim != 5:
            raise ValueError(
                "RA representation visualization expects [B,C,T,H,W], "
                f"got {tuple(pred_repr.shape)} and {tuple(target_repr.shape)}."
            )
        if pred_repr.shape != target_repr.shape:
            raise ValueError(
                "RA representation visualization shape mismatch: "
                f"pred={tuple(pred_repr.shape)} target={tuple(target_repr.shape)}."
            )

        _, channels, frames, height, width = [int(v) for v in pred_repr.shape]
        if channels <= 0 or frames <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"Invalid RA representation visualization shape: {tuple(pred_repr.shape)}")

        pred = pred_repr[0].detach().float().cpu()
        target = target_repr[0].detach().float().cpu()
        pred_tokens = pred.permute(1, 2, 3, 0).reshape(frames * height * width, channels)
        target_tokens = target.permute(1, 2, 3, 0).reshape(frames * height * width, channels)
        combined = torch.cat([target_tokens, pred_tokens], dim=0)
        combined = combined - combined.mean(dim=0, keepdim=True)
        q = min(3, int(combined.shape[0] - 1), int(combined.shape[1]))
        if q <= 0:
            raise ValueError(f"Cannot PCA-project RA representation tokens with shape {tuple(combined.shape)}.")
        _, _, v = torch.pca_lowrank(combined, q=q, center=False)
        projected = combined @ v[:, :q]
        if q < 3:
            projected = torch.cat(
                [projected, projected.new_zeros(projected.shape[0], 3 - q)],
                dim=1,
            )

        lo = projected.amin(dim=0, keepdim=True)
        hi = projected.amax(dim=0, keepdim=True)
        projected = ((projected - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
        token_count = frames * height * width
        target_rgb, pred_rgb = projected[:token_count], projected[token_count:]
        target_rgb = target_rgb.reshape(frames, height, width, 3).permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.reshape(frames, height, width, 3).permute(0, 3, 1, 2)
        diff_rgb = (pred_rgb - target_rgb).abs()

        rows = []
        for frame_idx in range(frames):
            rows.append(torch.cat([target_rgb[frame_idx], pred_rgb[frame_idx], diff_rgb[frame_idx]], dim=1))
        return torch.cat(rows, dim=2).contiguous()

    def _maybe_store_representation_visualization(self, pred_repr: torch.Tensor, target_repr: torch.Tensor):
        if not self._capture_representation_viz:
            return
        self._capture_representation_viz = False
        self._last_representation_viz = None
        with torch.no_grad():
            image = self._repr_latents_to_pca_rgb_strip(
                pred_repr=pred_repr,
                target_repr=target_repr,
            )
        self._last_representation_viz = {
            "tag": "train/representation_pca/ra_target_pred_diff",
            "image": image,
        }

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _merge_camera_codec_latents(self, camera_latents: list[torch.Tensor]) -> torch.Tensor:
        if not camera_latents:
            raise ValueError("Representation codec received no camera latents.")
        layout = str(getattr(self.representation_encoder, "camera_layout", "none")).lower()
        num_cameras = int(getattr(self.representation_encoder, "num_cameras", 1))
        ref_shape = camera_latents[0].shape[:3]
        for idx, latent in enumerate(camera_latents):
            if latent.ndim != 5 or latent.shape[:3] != ref_shape:
                raise ValueError(
                    "Camera codec latents must share [B,C,T], "
                    f"camera0={tuple(camera_latents[0].shape)} camera{idx}={tuple(latent.shape)}."
                )

        if layout in {"none", "single", "null"}:
            if len(camera_latents) != 1:
                raise ValueError(f"Single-camera codec layout expected 1 camera, got {len(camera_latents)}.")
            return camera_latents[0].contiguous()

        if layout == "horizontal":
            if len(camera_latents) != num_cameras:
                raise ValueError(
                    f"Horizontal codec layout expected {num_cameras} cameras, got {len(camera_latents)}."
                )
            heights = {int(latent.shape[-2]) for latent in camera_latents}
            if len(heights) != 1:
                raise ValueError(f"Horizontal camera latent heights must match, got {sorted(heights)}.")
            return torch.cat(camera_latents, dim=-1).contiguous()

        if layout == "vertical":
            if len(camera_latents) != num_cameras:
                raise ValueError(
                    f"Vertical codec layout expected {num_cameras} cameras, got {len(camera_latents)}."
                )
            widths = {int(latent.shape[-1]) for latent in camera_latents}
            if len(widths) != 1:
                raise ValueError(f"Vertical camera latent widths must match, got {sorted(widths)}.")
            return torch.cat(camera_latents, dim=-2).contiguous()

        if layout == "robotwin":
            if len(camera_latents) != 3 or num_cameras != 3:
                raise ValueError(
                    f"RobotWin codec layout requires exactly 3 cameras, got {len(camera_latents)}/{num_cameras}."
                )
            top, left, right = camera_latents
            if left.shape[-2:] != right.shape[-2:]:
                raise ValueError(
                    f"RobotWin wrist latent shapes must match, got {left.shape[-2:]} and {right.shape[-2:]}."
                )
            if int(top.shape[-2]) != 2 * int(left.shape[-2]):
                raise ValueError(
                    f"RobotWin front latent height must be twice wrist height, got {top.shape[-2:]} and {left.shape[-2:]}."
                )
            if int(top.shape[-1]) != int(left.shape[-1]) + int(right.shape[-1]):
                raise ValueError(
                    f"RobotWin front width must equal both wrist widths, got {top.shape[-2:]} and {left.shape[-2:]}."
                )
            return torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2).contiguous()

        raise ValueError(f"Unsupported representation camera layout for codec: {layout!r}.")

    @staticmethod
    def _camera_codec_module(module: nn.Module, camera_idx: int) -> nn.Module:
        if not isinstance(module, nn.ModuleDict):
            return module
        key = "front" if int(camera_idx) == 0 else "wrist"
        if key not in module:
            raise ValueError(f"Camera-specific codec is missing module {key!r}.")
        return module[key]

    @torch.no_grad()
    def _encode_camera_features(
        self,
        camera_videos: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not camera_videos:
            raise ValueError("Camera feature encoding received no camera videos.")
        batch_size = int(camera_videos[0].shape[0])
        resolution_groups: dict[tuple[int, int], list[int]] = {}
        for camera_idx, camera_video in enumerate(camera_videos):
            if camera_video.ndim != 5 or int(camera_video.shape[0]) != batch_size:
                raise ValueError(
                    f"Invalid camera video {camera_idx}: {tuple(camera_video.shape)}."
                )
            resolution = (
                int(camera_video.shape[-2]),
                int(camera_video.shape[-1]),
            )
            resolution_groups.setdefault(resolution, []).append(camera_idx)

        camera_features: list[Optional[torch.Tensor]] = [None] * len(camera_videos)
        for camera_indices in resolution_groups.values():
            packed_video = torch.cat(
                [camera_videos[idx] for idx in camera_indices],
                dim=0,
            )
            features = self.representation_encoder.forward_camera_dense(packed_video)
            if int(features.shape[1]) != self.encoder_output_dim:
                raise ValueError(
                    "Representation encoder feature dim mismatch before codec: "
                    f"got {features.shape[1]}, expected {self.encoder_output_dim}."
                )
            if self.normalize_target_mode != "none":
                features = self._normalize_representation_features(features)
            for camera_idx, feature in zip(
                camera_indices,
                features.split(batch_size, dim=0),
            ):
                camera_features[camera_idx] = feature.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )

        if any(feature is None for feature in camera_features):
            raise RuntimeError(
                "Failed to produce a DINO feature target for every camera."
            )
        return [feature for feature in camera_features if feature is not None]

    def _encode_camera_codec_latents(
        self,
        camera_videos: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.representation_codec is None:
            raise ValueError("Per-camera codec encoding requires an enabled representation codec.")
        if not camera_videos:
            raise ValueError("Per-camera codec encoding received no camera videos.")
        batch_size = int(camera_videos[0].shape[0])
        resolution_groups: dict[tuple[int, int], list[int]] = {}
        for camera_idx, camera_video in enumerate(camera_videos):
            if camera_video.ndim != 5 or int(camera_video.shape[0]) != batch_size:
                raise ValueError(f"Invalid camera video {camera_idx}: {tuple(camera_video.shape)}.")
            resolution = (int(camera_video.shape[-2]), int(camera_video.shape[-1]))
            resolution_groups.setdefault(resolution, []).append(camera_idx)

        camera_features: list[Optional[torch.Tensor]] = [None] * len(camera_videos)
        for camera_indices in resolution_groups.values():
            packed_video = torch.cat([camera_videos[idx] for idx in camera_indices], dim=0)
            with torch.no_grad():
                features = self.representation_encoder.forward_camera_dense(packed_video)
                if int(features.shape[1]) != self.encoder_output_dim:
                    raise ValueError(
                        "Representation encoder feature dim mismatch before codec: "
                        f"got {features.shape[1]}, expected {self.encoder_output_dim}."
                    )
                features = features.to(device=self.device, dtype=self.torch_dtype).detach()
            feature_chunks = features.split(batch_size, dim=0)
            for camera_idx, feature in zip(camera_indices, feature_chunks):
                camera_features[camera_idx] = feature.to(device=self.device, dtype=self.torch_dtype)

        if any(feature is None for feature in camera_features):
            raise RuntimeError("Failed to produce a codec feature target for every camera.")
        resolved_features = [feature for feature in camera_features if feature is not None]

        if isinstance(self.representation_codec, FAEAttentionCausalCodec):
            camera_latents = self.representation_codec(resolved_features)
        else:
            camera_latents_optional: list[Optional[torch.Tensor]] = [None] * len(camera_videos)
            for camera_indices in resolution_groups.values():
                codec_modules = {
                    id(self._camera_codec_module(self.representation_codec, idx))
                    for idx in camera_indices
                }
                if len(codec_modules) != 1:
                    raise ValueError(
                        "Cameras packed by resolution must share one codec encoder, "
                        f"got camera indices {camera_indices}."
                    )
                codec = self._camera_codec_module(self.representation_codec, camera_indices[0])
                packed_features = torch.cat(
                    [resolved_features[idx] for idx in camera_indices],
                    dim=0,
                )
                packed_latents = codec(packed_features)
                for camera_idx, latent in zip(
                    camera_indices,
                    packed_latents.split(batch_size, dim=0),
                ):
                    camera_latents_optional[camera_idx] = latent
            if any(latent is None for latent in camera_latents_optional):
                raise RuntimeError("Failed to produce a codec latent for every camera.")
            camera_latents = [
                latent for latent in camera_latents_optional if latent is not None
            ]

        return (
            camera_latents,
            resolved_features,
        )

    def _encode_codec_representation(
        self,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        camera_videos = self.representation_encoder.split_cameras(video)
        camera_latents, camera_features = self._encode_camera_codec_latents(
            camera_videos
        )
        latents = self._merge_camera_codec_latents(camera_latents)
        expected_shape = (
            int(video.shape[0]),
            self.target_dim,
            (
                (int(video.shape[2]) + 1) // 2
                if getattr(self, "representation_codec_temporal_padding", "left_zero") == "left_zero"
                else int(video.shape[2]) // 2
            ),
            int(self.latent_spatial_size[0]),
            int(self.latent_spatial_size[1]),
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                "Representation codec output shape mismatch: "
                f"got {tuple(latents.shape)}, expected {expected_shape}."
            )
        return latents.to(device=self.device, dtype=self.torch_dtype), camera_features

    def _normalize_codec_latents(
        self,
        raw_latents: torch.Tensor,
        *,
        update_stats: bool | None = None,
    ) -> torch.Tensor:
        codec_latent_norm = getattr(self, "codec_latent_norm", None)
        if codec_latent_norm is None:
            return raw_latents
        return codec_latent_norm(raw_latents, update_stats=update_stats)

    def _denormalize_codec_latents(self, latents: torch.Tensor) -> torch.Tensor:
        codec_latent_norm = getattr(self, "codec_latent_norm", None)
        if codec_latent_norm is None:
            return latents
        return codec_latent_norm.denormalize(latents)

    @torch.no_grad()
    def _encode_representation_latents(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,C,T,H,W], got {tuple(video.shape)}")
        if self.representation_codec is not None:
            raw_latents, _ = self._encode_codec_representation(video)
            return self._normalize_codec_latents(raw_latents, update_stats=False)

        camera_feature_transform = (
            self._normalize_representation_features
            if self.normalize_target_mode == "dataset"
            else None
        )
        features = self.representation_encoder.forward_pixels(
            video,
            camera_feature_transform=camera_feature_transform,
        )
        if features.ndim != 5:
            raise ValueError(
                "Representation encoder must return dense [B,C,T,H,W] features for RA, "
                f"got {tuple(features.shape)}."
            )
        if int(features.shape[1]) != self.encoder_output_dim:
            raise ValueError(
                f"Representation feature dim mismatch: got {features.shape[1]}, expected {self.encoder_output_dim}."
            )
        features = features.float()
        if self.latent_spatial_size is not None:
            features = F.interpolate(
                features,
                size=(int(features.shape[2]), int(self.latent_spatial_size[0]), int(self.latent_spatial_size[1])),
                mode="trilinear",
                align_corners=False,
            )
        if self.normalize_target_mode != "dataset":
            features = self._normalize_representation_features(features)
        return features.to(device=self.device, dtype=self.torch_dtype)

    def _align_representation_temporal_size(self, features: torch.Tensor, target_frames: int) -> torch.Tensor:
        target_frames = int(target_frames)
        if int(features.shape[2]) == target_frames:
            return features
        if self.temporal_align_mode == "strict":
            raise ValueError(
                "RA representation temporal mismatch: "
                f"encoder returned T={features.shape[2]}, expected T={target_frames}. "
                "Set `representation.temporal_align_mode=interpolate` to allow resizing."
            )
        return F.interpolate(
            features.float(),
            size=(target_frames, int(features.shape[-2]), int(features.shape[-1])),
            mode="trilinear",
            align_corners=False,
        ).to(device=features.device, dtype=features.dtype)

    def _select_training_video(self, video: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_frames = int(video.shape[2])
        if self.temporal_groups is not None:
            max_index = max(max(group) for group in self.temporal_groups)
            if max_index >= num_frames:
                raise ValueError(
                    "RA `representation.temporal_groups` exceeds sampled video length: "
                    f"groups={self.temporal_groups}, video_frames={num_frames}."
                )
            flat_indices = [idx for group in self.temporal_groups for idx in group]
            index_tensor = torch.as_tensor(flat_indices, device=video.device, dtype=torch.long)
            return video.index_select(dim=2, index=index_tensor), len(self.temporal_groups)
        if self.temporal_indices is not None:
            max_index = max(self.temporal_indices)
            if max_index >= num_frames:
                raise ValueError(
                    "RA `representation.temporal_indices` exceeds sampled video length: "
                    f"indices={self.temporal_indices}, video_frames={num_frames}."
                )
            index_tensor = torch.as_tensor(self.temporal_indices, device=video.device, dtype=torch.long)
            return video.index_select(dim=2, index=index_tensor), len(self.temporal_indices)
        return video, num_frames

    def _select_image_pad_mask(self, image_is_pad: torch.Tensor) -> torch.Tensor:
        if self.temporal_groups is not None:
            masks = []
            for group in self.temporal_groups:
                index_tensor = torch.as_tensor(group, device=image_is_pad.device, dtype=torch.long)
                # Match FastWAM's latent mask: a compressed temporal group is
                # padding only when every source frame is padding.
                masks.append(image_is_pad.index_select(dim=1, index=index_tensor).all(dim=1))
            return torch.stack(masks, dim=1)
        if self.temporal_indices is not None:
            index_tensor = torch.as_tensor(self.temporal_indices, device=image_is_pad.device, dtype=torch.long)
            return image_is_pad.index_select(dim=1, index=index_tensor)
        return image_is_pad

    def _prepare_context(
        self,
        *,
        prompt: Optional[Union[str, Sequence[str]]],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if context.shape[0] != batch_size:
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, -1, -1)
                context_mask = context_mask.expand(batch_size, -1)
            else:
                raise ValueError(f"Context batch mismatch: got {context.shape[0]}, expected {batch_size}.")
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        return context, context_mask

    def _prepare_codec_latents_for_training(
        self,
        online_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.representation_codec_trainable:
            return online_latents, None
        if int(online_latents.shape[2]) < 2:
            raise ValueError("Trainable representation codec requires current and future latents.")

        online_current = online_latents[:, :, :1]
        detached_future = online_latents[:, :, 1:].detach()
        if self.representation_codec_gradient_mode == "condition_and_action":
            return torch.cat((online_current, detached_future), dim=2), None
        if self.representation_codec_gradient_mode == "action_only":
            current_proxy = online_current.detach()
            if torch.is_grad_enabled():
                current_proxy = current_proxy.requires_grad_(True)
            return torch.cat((current_proxy, detached_future), dim=2), current_proxy
        raise ValueError(
            f"Unsupported codec gradient mode: {self.representation_codec_gradient_mode!r}."
        )

    def build_inputs(self, sample):
        video = sample["video"]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got {video.shape[1]}")
        batch_size, _, num_frames, _, _ = video.shape
        if num_frames <= 1:
            raise ValueError(f"`sample['video']` must contain at least 2 frames, got {num_frames}.")
        video, expected_repr_steps = self._select_training_video(video)
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for RA training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        raw_codec_latents = None
        if self.representation_codec is not None:
            raw_codec_latents, codec_feature_targets = self._encode_codec_representation(
                input_video
            )
            raw_codec_latents = self._align_representation_temporal_size(
                raw_codec_latents,
                expected_repr_steps,
            )
            # The RA wrapper stays in eval mode while selected trainable
            # submodules run in train mode. Let the normalization module's own
            # state decide whether lagged statistics should update.
            online_representation_latents = self._normalize_codec_latents(raw_codec_latents)
        else:
            online_representation_latents = self._encode_representation_latents(input_video)
            codec_feature_targets = []
            online_representation_latents = self._align_representation_temporal_size(
                online_representation_latents,
                expected_repr_steps,
            )
        representation_latents, codec_current_proxy = self._prepare_codec_latents_for_training(
            online_representation_latents
        )
        num_repr_steps = int(representation_latents.shape[2])
        if num_repr_steps <= 1:
            raise ValueError(f"RA representation latents must contain at least 2 steps, got {num_repr_steps}.")
        first_frame_latents = representation_latents[:, :, 0:1]

        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None or context_mask is None:
            prompt = sample.get("prompt")
        else:
            prompt = None
        proprio_seq = sample.get("proprio", None)
        if self.proprio_encoder is not None:
            if proprio_seq is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio_seq.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be [B,T,D], got {tuple(proprio_seq.shape)}")
            proprio = proprio_seq[:, 0, :]
        else:
            proprio = None
        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype) if proprio is not None else None,
            batch_size=batch_size,
        )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            image_is_pad = self._select_image_pad_mask(image_is_pad)
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "video": input_video,
            "representation_latents": representation_latents,
            "online_representation_latents": online_representation_latents,
            "raw_codec_latents": raw_codec_latents,
            "codec_current_proxy": codec_current_proxy,
            "codec_feature_targets": codec_feature_targets,
            "first_frame_latents": first_frame_latents,
            "action": action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _encode_inference_first_frame_latents(
        self,
        input_image: torch.Tensor,
        previous_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.representation_codec_uses_history:
            if previous_image is None:
                previous_image = input_image
            if previous_image.shape != input_image.shape:
                raise ValueError(
                    "`previous_image` must match `input_image`, got "
                    f"{tuple(previous_image.shape)} and {tuple(input_image.shape)}."
                )
            input_video = torch.stack((previous_image, input_image), dim=2)
            latents = self._encode_representation_latents(input_video)
            return self._align_representation_temporal_size(latents, 1)
        if self.temporal_groups is None:
            return self._encode_representation_latents(input_image.unsqueeze(2))
        first_group = self.temporal_groups[0]
        if any(idx != 0 for idx in first_group):
            raise ValueError(
                "RA inference only has the current image available, so the first temporal group must use frame 0 only; "
                f"got first group {first_group}."
            )
        input_video = input_image.unsqueeze(2).expand(-1, -1, len(first_group), -1, -1).contiguous()
        latents = self._encode_representation_latents(input_video)
        return self._align_representation_temporal_size(latents, 1)

    def _num_inference_representation_steps(self, num_video_frames: int) -> int:
        num_video_frames = int(num_video_frames)
        if num_video_frames <= 1:
            raise ValueError(
                f"`num_video_frames` must contain a current and future frame, got {num_video_frames}."
            )
        if self.temporal_groups is not None:
            max_index = max(max(group) for group in self.temporal_groups)
            if max_index >= num_video_frames:
                raise ValueError(
                    "RA `representation.temporal_groups` exceeds inference video length: "
                    f"groups={self.temporal_groups}, num_video_frames={num_video_frames}."
                )
            return len(self.temporal_groups)
        if self.temporal_indices is not None:
            max_index = max(self.temporal_indices)
            if max_index >= num_video_frames:
                raise ValueError(
                    "RA `representation.temporal_indices` exceeds inference video length: "
                    f"indices={self.temporal_indices}, num_video_frames={num_video_frames}."
                )
            return len(self.temporal_indices)
        return num_video_frames

    def validate_inference_timeline(
        self,
        *,
        num_video_frames: int,
        action_horizon: int,
    ) -> int:
        """Validate source-frame and action horizons against RA's latent timeline."""
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        num_repr_steps = self._num_inference_representation_steps(num_video_frames)
        num_future_repr_steps = num_repr_steps - 1
        if num_future_repr_steps <= 0:
            raise ValueError(
                "RA inference requires at least one future representation step, "
                f"got num_repr_steps={num_repr_steps}."
            )
        if self.mot_action_to_world.enabled and action_horizon % num_future_repr_steps != 0:
            raise ValueError(
                "Action horizon must be divisible by future representation steps when "
                "action-to-world attention is enabled, got "
                f"action_horizon={action_horizon}, future_repr_steps={num_future_repr_steps}."
            )
        return num_repr_steps

    def _split_representation_camera_latents(self, latents: torch.Tensor) -> list[torch.Tensor]:
        if latents.ndim != 5:
            raise ValueError(f"Representation latents must be [B,C,T,H,W], got {tuple(latents.shape)}")
        layout = str(getattr(self.representation_encoder, "camera_layout", "none")).lower()
        num_cameras = int(getattr(self.representation_encoder, "num_cameras", 1))
        height, width = int(latents.shape[-2]), int(latents.shape[-1])
        if layout in {"none", "single", "null"}:
            return [latents]
        if layout == "robotwin":
            if num_cameras != 3:
                raise ValueError(f"RobotWin decoder layout requires 3 cameras, got {num_cameras}.")
            top_height = max(1, int(round(height * 2.0 / 3.0)))
            if top_height >= height:
                raise ValueError(f"RobotWin latent is too short to split: H={height}.")
            left_width = max(1, width // 2)
            if left_width >= width:
                raise ValueError(f"RobotWin latent is too narrow to split: W={width}.")
            return [
                latents[..., :top_height, :],
                latents[..., top_height:, :left_width],
                latents[..., top_height:, left_width:],
            ]
        if layout == "horizontal":
            if width % num_cameras != 0:
                raise ValueError(
                    f"Horizontal latent width {width} is not divisible by num_cameras={num_cameras}."
                )
            camera_width = width // num_cameras
            return [latents[..., idx * camera_width : (idx + 1) * camera_width] for idx in range(num_cameras)]
        if layout == "vertical":
            if height % num_cameras != 0:
                raise ValueError(
                    f"Vertical latent height {height} is not divisible by num_cameras={num_cameras}."
                )
            camera_height = height // num_cameras
            return [latents[..., idx * camera_height : (idx + 1) * camera_height, :] for idx in range(num_cameras)]
        raise ValueError(f"Unsupported representation camera layout for RAEv2 decoding: {layout!r}.")

    def _decode_codec_feature_latents(self, latents: torch.Tensor) -> list[torch.Tensor]:
        if self.codec_decoder is None:
            raise ValueError("Frozen representation codec has no enabled online feature decoder.")
        camera_latents = self._split_representation_camera_latents(latents)
        batch_size = int(latents.shape[0])
        resolution_groups: dict[tuple[int, int], list[int]] = {}
        for camera_idx, camera_latent in enumerate(camera_latents):
            resolution = (int(camera_latent.shape[-2]), int(camera_latent.shape[-1]))
            resolution_groups.setdefault(resolution, []).append(camera_idx)

        decoded_features: list[Optional[torch.Tensor]] = [None] * len(camera_latents)
        for camera_indices in resolution_groups.values():
            packed_latents = torch.cat([camera_latents[idx] for idx in camera_indices], dim=0)
            decoder_modules = {
                id(self._camera_codec_module(self.codec_decoder, idx))
                for idx in camera_indices
            }
            if len(decoder_modules) != 1:
                raise ValueError(
                    "Cameras packed by resolution must share one codec decoder, "
                    f"got camera indices {camera_indices}."
                )
            decoder = self._camera_codec_module(self.codec_decoder, camera_indices[0])
            packed_features = decoder(packed_latents)
            chunks = packed_features.split(batch_size, dim=0)
            for camera_idx, decoded in zip(camera_indices, chunks):
                decoded_features[camera_idx] = decoded
        if any(features is None for features in decoded_features):
            raise RuntimeError("Failed to decode every camera codec latent to DINO features.")
        return [features for features in decoded_features if features is not None]

    def _codec_decoder_training_loss(
        self,
        *,
        representation_latents: torch.Tensor,
        feature_targets: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.codec_decoder is None or self.codec_decoder_loss_weight == 0.0:
            return representation_latents.new_zeros(())
        decoded_features = self._decode_codec_feature_latents(representation_latents.detach())
        if len(decoded_features) != len(feature_targets):
            raise ValueError(
                "Codec feature decoder camera count mismatch: "
                f"decoded={len(decoded_features)} target={len(feature_targets)}."
            )
        losses = []
        for camera_idx, (decoded, target) in enumerate(zip(decoded_features, feature_targets)):
            if decoded.shape != target.shape:
                raise ValueError(
                    f"Codec feature decoder camera {camera_idx} shape mismatch: "
                    f"decoded={tuple(decoded.shape)} target={tuple(target.shape)}."
                )
            losses.append(F.mse_loss(decoded.float(), target.detach().float()))
        return torch.stack(losses).mean()

    def has_codec_decoder(self) -> bool:
        return self.codec_decoder is not None

    def _strip_history_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        if bool(getattr(self, "representation_codec_uses_history", False)):
            if len(frames) < 2:
                raise ValueError("History-aware representation decoding returned fewer than two frames.")
            return frames[1:]
        return frames

    @torch.no_grad()
    def reconstruct_representation_video(self, video: torch.Tensor) -> list[Image.Image]:
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or int(video.shape[0]) != 1:
            raise ValueError(f"Representation reconstruction expects [1,3,T,H,W], got {tuple(video.shape)}.")
        selected_video, _ = self._select_training_video(video)
        selected_video = selected_video.to(device=self.device, dtype=self.torch_dtype)
        latents = self._encode_representation_latents(selected_video)
        return self._strip_history_frames(self._decode_representation_latents(latents))

    @torch.no_grad()
    def prepare_representation_rgb_target(self, video: torch.Tensor) -> list[Image.Image]:
        """Match GT camera layout and resolution to the released RAE decoder output."""
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or int(video.shape[0]) != 1:
            raise ValueError(f"Representation RGB target expects [1,3,T,H,W], got {tuple(video.shape)}.")
        selected_video, _ = self._select_training_video(video)
        camera_images = []
        for camera_video in self.representation_encoder.split_cameras(selected_video):
            batch, _, frames = camera_video.shape[:3]
            images = camera_video.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, camera_video.shape[-2], camera_video.shape[-1]
            )
            images = ((images.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
            camera_images.append(
                F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
            )
        target = self._merge_square_camera_rgb(camera_images, batch=batch, frames=frames)
        return self._strip_history_frames(self._video_tensor_to_pil(target))

    @staticmethod
    def _merge_square_camera_rgb(
        camera_images: list[torch.Tensor],
        *,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        if not camera_images:
            raise ValueError("No camera RGB images were provided.")
        camera_videos = []
        for images in camera_images:
            if images.ndim != 4 or int(images.shape[0]) != batch * frames:
                raise ValueError(
                    f"Camera RGB images must be [B*T,3,H,W], got {tuple(images.shape)}."
                )
            camera_videos.append(
                images.reshape(batch, frames, 3, images.shape[-2], images.shape[-1]).permute(0, 2, 1, 3, 4)
            )
        return torch.cat(camera_videos, dim=-1).contiguous()

    @staticmethod
    def _video_tensor_to_pil(video: torch.Tensor) -> list[Image.Image]:
        if video.ndim != 5 or int(video.shape[0]) != 1 or int(video.shape[1]) != 3:
            raise ValueError(f"Decoded RGB video must be [1,3,T,H,W], got {tuple(video.shape)}.")
        frames = video[0].permute(1, 2, 3, 0).mul(255.0).round().byte().cpu().numpy()
        return [Image.fromarray(frame, mode="RGB") for frame in frames]

    def _denormalize_released_camera_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.normalize_target_mode != "dataset":
            raise ValueError("RAEv2 released decoder requires representation.normalize_target='dataset'.")
        if self.latent_mean is None or self.latent_var is None:
            raise ValueError("RAEv2 released decoder requires loaded released dataset statistics.")
        mean = self.latent_mean.to(device=latents.device, dtype=torch.float32)
        var = self.latent_var.to(device=latents.device, dtype=torch.float32)
        while mean.ndim > 3 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
            var = var.squeeze(0)
        if mean.shape != (self.encoder_output_dim, 16, 16) or var.shape != mean.shape:
            raise ValueError(
                "RAEv2 released decoder requires [1024,16,16] mean/var, "
                f"got mean={tuple(mean.shape)} var={tuple(var.shape)}."
            )
        return (
            latents.float() * torch.sqrt(var.unsqueeze(0).clamp_min(0.0) + self.latent_stats_eps)
            + mean.unsqueeze(0)
        ).to(dtype=self.torch_dtype)

    def _get_representation_decoder(self):
        if self.representation_decoder is None:
            if self.representation_decoder_path is None:
                raise ValueError("Set representation.decoder_path to the released RAEv2 decoder.pt.")
            from fastwam.models.raev2_decoder import RAEv2GeneralDecoder

            self.representation_decoder = RAEv2GeneralDecoder.from_checkpoint(
                self.representation_decoder_path,
                device=self.device,
                dtype=self.torch_dtype,
            )
            logger.info("Loaded RAEv2 released decoder from %s.", self.representation_decoder_path)
        return self.representation_decoder

    def release_representation_decoder(self):
        """Drop the frozen RAEv2 visualization decoder before training resumes."""
        self.representation_decoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _decode_camera_features_to_rgb_tensor(
        self,
        camera_features: list[torch.Tensor],
    ) -> torch.Tensor:
        decoder = self._get_representation_decoder()
        decoder_parameter = next(decoder.parameters(), None)
        decoded_cameras = []
        batch = frames = None
        for features in camera_features:
            camera_batch, channels, camera_frames = features.shape[:3]
            if batch is None:
                batch, frames = int(camera_batch), int(camera_frames)
            elif (int(camera_batch), int(camera_frames)) != (batch, frames):
                raise ValueError("All camera feature tensors must share batch and temporal dimensions.")
            features = features.permute(0, 2, 1, 3, 4).reshape(
                camera_batch * camera_frames,
                channels,
                features.shape[-2],
                features.shape[-1],
            )
            features = F.interpolate(
                features.float(),
                size=(16, 16),
                mode="bilinear",
                align_corners=False,
            )
            # Legacy codec-free RAE paths may still expose dataset-normalized
            # features. Codec-based RARAE now reconstructs raw DINO MLS features.
            if self.normalize_target_mode == "dataset":
                features = self._denormalize_released_camera_latents(features)
            # The visualization decoder is loaded lazily after Accelerate has
            # prepared the trainable model. Match its actual parameter dtype and
            # device at the call boundary instead of assuming self.torch_dtype is
            # still authoritative (or relying on an outer autocast context).
            if decoder_parameter is not None:
                features = features.to(
                    device=decoder_parameter.device,
                    dtype=decoder_parameter.dtype,
                )
            decoded_cameras.append(decoder(features).float().clamp(0.0, 1.0))
        if batch is None or frames is None:
            raise ValueError("No camera DINO features were provided for RGB decoding.")
        return self._merge_square_camera_rgb(decoded_cameras, batch=batch, frames=frames)

    @torch.no_grad()
    def _decode_representation_latents(self, latents: torch.Tensor) -> list[Image.Image]:
        if self.representation_codec is not None:
            raw_latents = self._denormalize_codec_latents(latents)
            camera_features = self._decode_codec_feature_latents(raw_latents)
        else:
            camera_features = self._split_representation_camera_latents(latents)
        decoded = self._decode_camera_features_to_rgb_tensor(camera_features)
        return self._video_tensor_to_pil(decoded)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_world_action_mot_mask(
            world_expert=self.representation_expert,
            world_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            world_tokens_per_frame=video_tokens_per_frame,
            device=device,
            action_to_world=self.mot_action_to_world,
        )

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        *,
        latents_repr: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_repr: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(getattr(self.representation_expert, "action_conditioned", False)):
            raise ValueError("This RARAE rollout path currently supports action_conditioned=false only.")
        if self.mot_action_to_world_enabled:
            raise ValueError("This RARAE rollout path currently supports action_to_world=false only.")
        repr_pre = self.representation_expert.pre_dit(
            x=latents_repr,
            timestep=timestep_repr,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=int(repr_pre["tokens"].shape[1]),
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(repr_pre["meta"]["tokens_per_frame"]),
            device=repr_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": repr_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": repr_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": repr_pre["context"], "mask": repr_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": repr_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return (
            self.representation_expert.post_dit(tokens_out["video"], repr_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    def _compute_representation_loss_per_sample(
        self,
        pred_repr: torch.Tensor,
        target_repr: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_step: bool,
    ) -> torch.Tensor:
        loss_token = F.mse_loss(pred_repr.float(), target_repr.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return loss_token.mean(dim=1)
        if image_is_pad.shape[1] != loss_token.shape[1] + (0 if include_initial_step else 1):
            raise ValueError(
                "Representation-loss mask shape mismatch: "
                f"mask steps={image_is_pad.shape[1]}, loss steps={loss_token.shape[1]}, include_initial={include_initial_step}."
            )
        repr_is_pad = image_is_pad if include_initial_step else image_is_pad[:, 1:]
        valid = (~repr_is_pad).to(device=loss_token.device, dtype=loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (loss_token * valid).sum(dim=1) / valid_sum

    def _sigma_from_timestep(self, timestep: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.train_representation_scheduler.num_train_timesteps)).to(
            device=target.device,
            dtype=target.dtype,
        )
        if sigma.ndim == 0:
            return sigma
        if sigma.ndim == 1:
            if sigma.shape[0] not in (1, target.shape[0]):
                raise ValueError(
                    "Batch timestep shape mismatch: "
                    f"timestep={tuple(timestep.shape)}, target={tuple(target.shape)}."
                )
            return sigma.view(-1, *([1] * (target.ndim - 1)))
        if sigma.ndim == 2:
            if target.ndim < 3 or sigma.shape != (target.shape[0], target.shape[2]):
                raise ValueError(
                    "Per-frame timestep must match target [B, C, T, ...]: "
                    f"timestep={tuple(timestep.shape)}, target={tuple(target.shape)}."
                )
            return sigma.view(target.shape[0], 1, target.shape[2], *([1] * (target.ndim - 3)))
        raise ValueError(f"Unsupported timestep shape: {tuple(timestep.shape)}.")

    def _sample_representation_training_timestep(
        self,
        representation_latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(representation_latents.shape[0])
        if self.representation_noise_timestep_mode == "shared":
            return self.train_representation_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=representation_latents.dtype,
            )

        num_frames = int(representation_latents.shape[2])
        if num_frames < 2:
            raise ValueError(
                "Per-frame representation timesteps require at least one observation "
                "and one future latent frame."
            )
        future_timestep = self.train_representation_scheduler.sample_training_t(
            batch_size=batch_size * (num_frames - 1),
            device=self.device,
            dtype=representation_latents.dtype,
        ).reshape(batch_size, num_frames - 1)
        clean_observation_timestep = torch.zeros(
            (batch_size, 1),
            device=future_timestep.device,
            dtype=future_timestep.dtype,
        )
        return torch.cat((clean_observation_timestep, future_timestep), dim=1)

    def _representation_training_target(
        self,
        *,
        representation_latents: torch.Tensor,
        noise_repr: torch.Tensor,
        noisy_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
    ) -> torch.Tensor:
        del noisy_repr
        return self.train_representation_scheduler.training_target(
            representation_latents,
            noise_repr,
            timestep_repr,
        )

    def _representation_diffusion_latents(self, representation_latents: torch.Tensor) -> torch.Tensor:
        if self.representation_state_space == "absolute":
            return representation_latents
        current_repr = representation_latents[:, :, 0:1]
        return representation_latents - current_repr

    def _prediction_to_clean_state(
        self,
        *,
        pred_repr: torch.Tensor,
        noisy_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self._sigma_from_timestep(timestep_repr, pred_repr)
        return noisy_repr - sigma * pred_repr

    def _compute_representation_velocity_loss_per_sample(
        self,
        *,
        pred_repr: torch.Tensor,
        target_repr: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._compute_representation_loss_per_sample(
            pred_repr=pred_repr,
            target_repr=target_repr,
            image_is_pad=image_is_pad,
            include_initial_step=False,
        )

    @staticmethod
    def _safe_scalar(value: torch.Tensor) -> float:
        return float(value.detach().float().mean().item())

    def _representation_monitor_metrics(
        self,
        *,
        representation_latents: torch.Tensor,
        clean_future: torch.Tensor,
        noisy_future: torch.Tensor,
        noise_future: torch.Tensor,
        pred_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
        loss_repr_raw: torch.Tensor,
        loss_repr: torch.Tensor,
        absolute_future: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        future_timestep = timestep_repr[:, 1:] if timestep_repr.ndim == 2 else timestep_repr
        metrics = {
            "repr/target_mean": self._safe_scalar(clean_future.mean()),
            "repr/target_std": self._safe_scalar(clean_future.std()),
            "repr/target_norm": self._safe_scalar(clean_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/noise_norm": self._safe_scalar(noise_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/noisy_norm": self._safe_scalar(noisy_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/pred_norm": self._safe_scalar(pred_repr.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/loss_raw": float(loss_repr_raw.detach().float().item()),
            "repr/loss_weighted": float(loss_repr.detach().float().item()),
            "repr/sigma_mean": self._safe_scalar(
                future_timestep.float()
                / float(self.train_representation_scheduler.num_train_timesteps)
            ),
            "repr/shift": float(self.train_representation_scheduler.shift),
            "repr/state_space_delta": float(self.representation_state_space == "delta"),
            "repr/per_frame_timestep": float(self.representation_noise_timestep_mode == "per_frame"),
        }
        if timestep_repr.ndim == 2 and timestep_repr.shape[1] > 1:
            future_sigma = future_timestep.float() / float(
                self.train_representation_scheduler.num_train_timesteps
            )
            metrics["repr/sigma_future_std"] = self._safe_scalar(
                future_sigma.std(dim=1, unbiased=False)
            )
        if int(representation_latents.shape[2]) > 1:
            delta = representation_latents[:, :, 1:] - representation_latents[:, :, :-1]
            metrics["repr/delta_norm"] = self._safe_scalar(delta.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt())
        if absolute_future is not None and self.representation_state_space == "delta":
            metrics["repr/absolute_target_norm"] = self._safe_scalar(
                absolute_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
            )
        return metrics

    def training_loss(self, sample, tiled: bool = False):
        del tiled
        inputs = self.build_inputs(sample)
        representation_latents = inputs["representation_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        representation_diffusion_latents = self._representation_diffusion_latents(representation_latents)
        noise_repr = torch.randn_like(representation_diffusion_latents)
        batch_size = int(representation_latents.shape[0])
        timestep_repr = self._sample_representation_training_timestep(
            representation_diffusion_latents
        )
        noisy_repr = self.train_representation_scheduler.add_noise(
            representation_diffusion_latents,
            noise_repr,
            timestep_repr,
        )
        target_repr = self._representation_training_target(
            representation_latents=representation_diffusion_latents,
            noise_repr=noise_repr,
            noisy_repr=noisy_repr,
            timestep_repr=timestep_repr,
        )
        # Keep the observation token absolute even for delta-state prediction; the
        # future tokens carry the noisy delta state.
        noisy_repr[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        repr_pre = self.representation_expert.pre_dit(
            x=noisy_repr,
            timestep=timestep_repr,
            context=context,
            context_mask=context_mask,
            action=action if bool(getattr(self.representation_expert, "action_conditioned", False)) else None,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=repr_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(repr_pre["meta"]["tokens_per_frame"]),
            device=repr_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": repr_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": repr_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": repr_pre["context"],
                    "mask": repr_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": repr_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_repr = self.representation_expert.post_dit(tokens_out["video"], repr_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_repr = pred_repr[:, :, 1:]
        target_repr = target_repr[:, :, 1:]
        clean_future = representation_diffusion_latents[:, :, 1:]
        noisy_future = noisy_repr[:, :, 1:]
        noise_future = noise_repr[:, :, 1:]
        absolute_future = None
        if self.representation_state_space == "delta":
            absolute_future = representation_latents[:, :, 1:]
        if self._capture_representation_viz:
            future_timestep_repr = (
                timestep_repr[:, 1:] if timestep_repr.ndim == 2 else timestep_repr
            )
            clean_pred_for_viz = self._prediction_to_clean_state(
                pred_repr=pred_repr,
                noisy_repr=noisy_future,
                timestep_repr=future_timestep_repr,
            )
            if self.representation_state_space == "delta":
                viz_pred = representation_latents[:, :, 0:1] + clean_pred_for_viz
                viz_target = absolute_future
            else:
                viz_pred = clean_pred_for_viz
                viz_target = clean_future
            self._maybe_store_representation_visualization(
                pred_repr=viz_pred,
                target_repr=viz_target,
            )
        if timestep_repr.ndim == 1:
            loss_repr_per_sample = self._compute_representation_velocity_loss_per_sample(
                pred_repr=pred_repr,
                target_repr=target_repr,
                image_is_pad=image_is_pad,
            )
            repr_weight = self.train_representation_scheduler.training_weight(timestep_repr).to(
                loss_repr_per_sample.device, dtype=loss_repr_per_sample.dtype
            )
            loss_repr_raw = loss_repr_per_sample.mean()
            loss_repr = (loss_repr_per_sample * repr_weight).mean()
        else:
            loss_repr_per_frame = F.mse_loss(
                pred_repr.float(), target_repr.float(), reduction="none"
            ).mean(dim=(1, 3, 4))
            if image_is_pad is None:
                valid_repr = torch.ones_like(loss_repr_per_frame)
            else:
                if image_is_pad.shape[1] != loss_repr_per_frame.shape[1] + 1:
                    raise ValueError(
                        "Representation-loss mask shape mismatch: "
                        f"mask steps={image_is_pad.shape[1]}, "
                        f"loss steps={loss_repr_per_frame.shape[1]}."
                    )
                valid_repr = (~image_is_pad[:, 1:]).to(
                    device=loss_repr_per_frame.device,
                    dtype=loss_repr_per_frame.dtype,
                )
            valid_count = valid_repr.sum(dim=1).clamp(min=1.0)
            loss_repr_per_sample = (
                loss_repr_per_frame * valid_repr
            ).sum(dim=1) / valid_count
            repr_weight = self.train_representation_scheduler.training_weight(
                timestep_repr[:, 1:]
            ).to(loss_repr_per_frame.device, dtype=loss_repr_per_frame.dtype)
            weighted_loss_repr_per_sample = (
                loss_repr_per_frame * repr_weight * valid_repr
            ).sum(dim=1) / valid_count
            loss_repr_raw = loss_repr_per_sample.mean()
            loss_repr = weighted_loss_repr_per_sample.mean()

        loss_action = compute_action_flow_loss(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
            action_scheduler=self.train_action_scheduler,
        )

        loss_codec_route = loss_action.new_zeros(())
        codec_action_proxy_grad_norm = None
        codec_current_proxy = inputs["codec_current_proxy"]
        if (
            self.representation_codec_trainable
            and self.representation_codec_gradient_mode == "action_only"
            and torch.is_grad_enabled()
        ):
            if codec_current_proxy is None:
                raise RuntimeError("Action-only codec routing is missing its current-latent proxy.")
            codec_action_gradient = torch.autograd.grad(
                self.loss_lambda_action * loss_action,
                codec_current_proxy,
                retain_graph=True,
                create_graph=False,
            )[0]
            online_current = inputs["online_representation_latents"][:, :, :1]
            loss_codec_route = (
                (online_current - online_current.detach())
                * codec_action_gradient.detach()
            ).sum()
            codec_action_proxy_grad_norm = self._safe_scalar(
                codec_action_gradient.float().pow(2).mean().sqrt()
            )

        loss_decoder_raw = self._codec_decoder_training_loss(
            representation_latents=(
                inputs["raw_codec_latents"]
                if inputs["raw_codec_latents"] is not None
                else representation_latents
            ),
            feature_targets=inputs["codec_feature_targets"],
        )
        loss_decoder = self.codec_decoder_loss_weight * loss_decoder_raw

        loss_total = (
            self.loss_lambda_representation * loss_repr
            + self.loss_lambda_action * loss_action
            + loss_decoder
            + loss_codec_route
        )
        loss_dict = {
            "loss_representation": self.loss_lambda_representation * float(loss_repr.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_codec_decoder": float(loss_decoder.detach().item()),
            "conditioning/action_enabled": float(
                bool(getattr(self.representation_expert, "action_conditioned", False))
            ),
            "conditioning/mot_action_to_world_enabled": float(self.mot_action_to_world_enabled),
            "repr/codec_enabled": float(self.representation_codec_enabled),
            "repr/codec_trainable": float(self.representation_codec_trainable),
            "repr/codec_condition_gradient": float(
                self.representation_codec_gradient_mode == "condition_and_action"
            ),
            "repr/codec_action_only_gradient": float(
                self.representation_codec_gradient_mode == "action_only"
            ),
            "repr/codec_dim": float(self.target_dim),
        }
        if codec_action_proxy_grad_norm is not None:
            loss_dict["repr/codec_action_proxy_grad_rms"] = codec_action_proxy_grad_norm
        codec_attention_entropy = getattr(
            self.representation_codec,
            "last_attention_entropy",
            None,
        )
        if codec_attention_entropy is not None:
            loss_dict["repr/codec_attention_entropy"] = self._safe_scalar(
                codec_attention_entropy
            )
        raw_codec_latents = inputs["raw_codec_latents"]
        if raw_codec_latents is not None:
            loss_dict["repr/codec_raw_mean"] = self._safe_scalar(raw_codec_latents.mean())
            loss_dict["repr/codec_raw_std"] = self._safe_scalar(raw_codec_latents.std())
        if self.codec_latent_norm is not None:
            loss_dict["repr/codec_norm_running_mean_abs"] = self._safe_scalar(
                self.codec_latent_norm.running_mean.abs().mean()
            )
            loss_dict["repr/codec_norm_running_std"] = self._safe_scalar(
                self.codec_latent_norm.running_var.clamp_min(0.0).sqrt().mean()
            )
            loss_dict["repr/codec_norm_batches"] = float(
                self.codec_latent_norm.num_batches_tracked.item()
            )
        loss_dict.update(
            self._representation_monitor_metrics(
                representation_latents=representation_latents,
                clean_future=clean_future,
                noisy_future=noisy_future,
                noise_future=noise_future,
                pred_repr=pred_repr,
                timestep_repr=timestep_repr,
                loss_repr_raw=loss_repr_raw,
                loss_repr=loss_repr,
                absolute_future=absolute_future,
            )
        )
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        previous_image: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, tiled
        self.eval()
        num_repr_steps = self.validate_inference_timeline(
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
        )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if previous_image is not None:
            if previous_image.ndim == 3:
                previous_image = previous_image.unsqueeze(0)
            if previous_image.shape != input_image.shape:
                raise ValueError(
                    "`previous_image` must match `input_image`, got "
                    f"{tuple(previous_image.shape)} and {tuple(input_image.shape)}."
                )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            batch_size=1,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if previous_image is not None:
            previous_image = previous_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_repr = self._encode_inference_first_frame_latents(
            input_image,
            previous_image=previous_image,
        )
        _, repr_dim, _, repr_h, repr_w = first_frame_repr.shape

        repr_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_repr = torch.randn(
            (1, repr_dim, num_repr_steps, repr_h, repr_w),
            generator=repr_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_repr[:, :, 0:1] = first_frame_repr

        infer_timesteps_repr, infer_deltas_repr = self.infer_representation_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_repr.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        if len(infer_timesteps_repr) != len(infer_timesteps_action):
            raise ValueError("Representation/action inference schedules must have the same number of steps.")
        for step_t_repr, step_delta_repr, step_t_action, step_delta_action in zip(
            infer_timesteps_repr,
            infer_deltas_repr,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            pred_repr, pred_action = self._predict_joint_noise(
                latents_repr=latents_repr,
                latents_action=latents_action,
                timestep_repr=step_t_repr.unsqueeze(0).to(device=self.device, dtype=latents_repr.dtype),
                timestep_action=step_t_action.unsqueeze(0).to(device=self.device, dtype=latents_action.dtype),
                context=context,
                context_mask=context_mask,
            )
            latents_repr = self.infer_representation_scheduler.step(
                pred_repr,
                step_delta_repr,
                latents_repr,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action,
                step_delta_action,
                latents_action,
            )
            latents_repr[:, :, 0:1] = first_frame_repr

        decode_latents = latents_repr
        if self.representation_state_space == "delta":
            decode_latents = latents_repr.clone()
            decode_latents[:, :, 1:] = first_frame_repr + latents_repr[:, :, 1:]
        return {
            "video": self._strip_history_frames(
                self._decode_representation_latents(decode_latents)
            ),
            "representation": decode_latents.detach().to(device="cpu", dtype=torch.float32),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        previous_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: Optional[int] = None,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, tiled, num_video_frames
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        if previous_image is not None:
            if previous_image.ndim == 3:
                previous_image = previous_image.unsqueeze(0)
            if previous_image.shape != input_image.shape:
                raise ValueError(
                    "`previous_image` must match `input_image`, got "
                    f"{tuple(previous_image.shape)} and {tuple(input_image.shape)}."
                )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            batch_size=1,
        )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if previous_image is not None:
            previous_image = previous_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_repr = self._encode_inference_first_frame_latents(
            input_image,
            previous_image=previous_image,
        )
        timestep_repr = torch.zeros((1,), dtype=first_frame_repr.dtype, device=self.device)
        repr_pre = self.representation_expert.pre_dit(
            x=first_frame_repr,
            timestep=timestep_repr,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        video_seq_len = int(repr_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(repr_pre["meta"]["tokens_per_frame"]),
            device=repr_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=repr_pre["tokens"],
            video_freqs=repr_pre["freqs"],
            video_t_mod=repr_pre["t_mod"],
            video_context_payload={
                "context": repr_pre["context"],
                "mask": repr_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        previous_image: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del action, action_cfg_scale
        if action_horizon is None:
            raise ValueError("RA.infer requires `action_horizon` because it only produces action output.")
        return self.infer_action(
            prompt=prompt,
            input_image=input_image,
            previous_image=previous_image,
            action_horizon=int(action_horizon),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            num_video_frames=num_frames,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "model_class": self.__class__.__name__,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.representation_codec is not None:
            payload["representation_codec"] = self.representation_codec.state_dict()
        if self.codec_latent_norm is not None:
            payload["codec_latent_norm"] = self.codec_latent_norm.state_dict()
        if self.codec_decoder is not None:
            payload["codec_decoder"] = self.codec_decoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" not in payload:
            raise ValueError(f"RA checkpoint missing `mot` key: {path}")
        self.mot.load_state_dict(payload["mot"], strict=True)
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder`, but current model disables it; ignoring.")
        if self.representation_codec is not None:
            if "representation_codec" in payload:
                self.representation_codec.load_state_dict(payload["representation_codec"], strict=True)
            else:
                logger.warning(
                    "Checkpoint has no `representation_codec` state; keeping the deterministic codec "
                    "initialized from the current config seed."
                )
        elif "representation_codec" in payload:
            logger.warning("Checkpoint contains `representation_codec`, but current model disables it; ignoring.")
        if self.codec_latent_norm is not None:
            if "codec_latent_norm" in payload:
                self.codec_latent_norm.load_state_dict(payload["codec_latent_norm"], strict=True)
            else:
                logger.warning(
                    "Checkpoint has no `codec_latent_norm` state. Keeping identity running "
                    "statistics; legacy checkpoints used a different pre-codec/post-codec "
                    "normalization geometry and are not exact training resumes."
                )
        elif "codec_latent_norm" in payload:
            logger.warning(
                "Checkpoint contains `codec_latent_norm`, but current model disables it; ignoring."
            )
        if self.codec_decoder is not None:
            if "codec_decoder" in payload:
                self.codec_decoder.load_state_dict(payload["codec_decoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `codec_decoder` weights; keeping the initialized decoder.")
        elif "codec_decoder" in payload:
            logger.warning(
                "Checkpoint contains `codec_decoder`, but current model "
                "disables it; ignoring."
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
