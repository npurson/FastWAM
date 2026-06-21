from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.action_dit import ActionDiT
from fastwam.models.helpers.io import ModelConfig, load_state_dict
from fastwam.models.mot import MoT
from fastwam.models.representation_encoders import build_representation_encoder
from fastwam.models.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return dict(value)


def _as_hw(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value in (None, "", "null"):
        return default
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"Expected 2D spatial size, got {value}.")
    return int(value[0]), int(value[1])


def _as_optional_path(value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    return str(value)


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
        representation_prediction: Optional[dict[str, Any]] = None,
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
        self.capability = {
            "video_inference": False,
            "vae_reconstruction": False,
            "action_inference": True,
        }
        self.capabilities = self.capability
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

        representation = _as_plain_dict(representation)
        representation_prediction = _as_plain_dict(representation_prediction)
        self.representation_prediction_type = str(
            representation_prediction.get("type", representation_prediction.get("prediction", "velocity"))
        ).lower()
        if self.representation_prediction_type not in {"velocity", "x"}:
            raise ValueError(
                "representation_prediction.type must be one of {'velocity', 'x'}, "
                f"got {self.representation_prediction_type!r}."
            )
        self.representation_x_loss_mode = str(representation_prediction.get("x_loss_mode", "direct")).lower()
        if self.representation_x_loss_mode not in {"direct", "velocity_equivalent"}:
            raise ValueError(
                "representation_prediction.x_loss_mode must be one of {'direct', 'velocity_equivalent'}, "
                f"got {self.representation_x_loss_mode!r}."
            )
        self.representation_t_eps = float(representation_prediction.get("t_eps", 0.05))
        if self.representation_t_eps <= 0:
            raise ValueError(f"representation_prediction.t_eps must be positive, got {self.representation_t_eps}.")

        self.target_dim = int(
            representation.get(
                "target_dim",
                getattr(self.representation_encoder, "output_dim", getattr(self.representation_expert, "in_dim", 2048)),
            )
        )
        if int(getattr(self.representation_expert, "in_dim", self.target_dim)) != self.target_dim:
            raise ValueError(
                "Representation expert `in_dim` must match representation target dim: "
                f"expert={getattr(self.representation_expert, 'in_dim', None)} target={self.target_dim}."
            )
        self.latent_spatial_size = _as_hw(representation.get("latent_spatial_size", None), default=(12, 10))
        normalize_target = representation.get("normalize_target", False)
        if isinstance(normalize_target, str):
            self.normalize_target_mode = normalize_target.lower()
        else:
            self.normalize_target_mode = "per_sample" if bool(normalize_target) else "none"
        if self.normalize_target_mode in {"false", "off", "no", "0"}:
            self.normalize_target_mode = "none"
        if self.normalize_target_mode in {"true", "on", "yes", "1"}:
            self.normalize_target_mode = "per_sample"
        if self.normalize_target_mode not in {"none", "per_sample", "dataset"}:
            raise ValueError(
                "RA representation.normalize_target must be false/true or one of "
                "{'none', 'per_sample', 'dataset'}, "
                f"got {normalize_target!r}."
            )
        self.normalize_target = self.normalize_target_mode != "none"
        self.latent_stats_path = _as_optional_path(
            representation.get("latent_stats_path", representation.get("normalization_stat_path", None))
        )
        self.register_buffer("latent_mean", None, persistent=False)
        self.register_buffer("latent_var", None, persistent=False)
        self.latent_stats_eps = float(representation.get("latent_stats_eps", 1e-5))
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
        self.temporal_align_mode = str(representation.get("temporal_align_mode", "strict")).lower()
        if self.temporal_align_mode not in {"strict", "interpolate"}:
            raise ValueError(f"Unsupported RA representation.temporal_align_mode: {self.temporal_align_mode}")
        self.temporal_groups = _parse_temporal_groups(
            representation.get("temporal_groups", None),
            "RA representation.temporal_groups",
            min_steps=2,
        )
        temporal_indices = representation.get(
            "temporal_indices",
            None if self.temporal_groups is not None else [0, 4, 8],
        )
        self.temporal_indices = _parse_temporal_indices(
            temporal_indices,
            "RA representation.temporal_indices",
            min_steps=2,
        )
        if self.temporal_groups is not None and self.temporal_indices is not None:
            raise ValueError("Use only one of RA `representation.temporal_indices` and `representation.temporal_groups`.")

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_representation = float(loss_lambda_representation)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)
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
        representation_prediction: Optional[dict[str, Any]] = None,
    ) -> "RA":
        representation = _as_plain_dict(representation)
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
            representation_prediction=representation_prediction,
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

    @torch.no_grad()
    def _encode_representation_latents(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,C,T,H,W], got {tuple(video.shape)}")
        features = self.representation_encoder.forward_pixels(video)
        if features.ndim != 5:
            raise ValueError(
                "Representation encoder must return dense [B,C,T,H,W] features for RA, "
                f"got {tuple(features.shape)}."
            )
        if int(features.shape[1]) != self.target_dim:
            raise ValueError(f"Representation feature dim mismatch: got {features.shape[1]}, expected {self.target_dim}.")
        features = features.float()
        if self.latent_spatial_size is not None:
            features = F.interpolate(
                features,
                size=(int(features.shape[2]), int(self.latent_spatial_size[0]), int(self.latent_spatial_size[1])),
                mode="trilinear",
                align_corners=False,
            )
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
                masks.append(image_is_pad.index_select(dim=1, index=index_tensor).any(dim=1))
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
        representation_latents = self._encode_representation_latents(input_video)
        representation_latents = self._align_representation_temporal_size(representation_latents, expected_repr_steps)
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
            "representation_latents": representation_latents,
            "first_frame_latents": first_frame_latents,
            "action": action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _encode_inference_first_frame_latents(self, input_image: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.representation_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

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
        return sigma.view(-1, *([1] * (target.ndim - 1)))

    def _representation_training_target(
        self,
        *,
        representation_latents: torch.Tensor,
        noise_repr: torch.Tensor,
        noisy_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
    ) -> torch.Tensor:
        if self.representation_prediction_type == "velocity":
            return self.train_representation_scheduler.training_target(
                representation_latents,
                noise_repr,
                timestep_repr,
            )
        del noise_repr, noisy_repr, timestep_repr
        return representation_latents

    def _compute_representation_prediction_loss_per_sample(
        self,
        *,
        pred_repr: torch.Tensor,
        target_repr: torch.Tensor,
        clean_repr: torch.Tensor,
        noisy_repr: torch.Tensor,
        noise_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.representation_prediction_type == "x" and self.representation_x_loss_mode == "velocity_equivalent":
            sigma = self._sigma_from_timestep(timestep_repr, pred_repr).clamp_min(self.representation_t_eps)
            pred_for_loss = (noisy_repr - pred_repr) / sigma
            target_for_loss = noise_repr - clean_repr
        else:
            pred_for_loss = pred_repr
            target_for_loss = target_repr
        return self._compute_representation_loss_per_sample(
            pred_repr=pred_for_loss,
            target_repr=target_for_loss,
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
        target_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
        loss_repr_raw: torch.Tensor,
        loss_repr: torch.Tensor,
    ) -> dict[str, float]:
        metrics = {
            "repr/target_mean": self._safe_scalar(clean_future.mean()),
            "repr/target_std": self._safe_scalar(clean_future.std()),
            "repr/target_norm": self._safe_scalar(clean_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/noise_norm": self._safe_scalar(noise_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/noisy_norm": self._safe_scalar(noisy_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/pred_norm": self._safe_scalar(pred_repr.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/loss_raw": float(loss_repr_raw.detach().float().item()),
            "repr/loss_weighted": float(loss_repr.detach().float().item()),
            "repr/sigma_mean": self._safe_scalar(timestep_repr.float() / float(self.train_representation_scheduler.num_train_timesteps)),
            "repr/shift": float(self.train_representation_scheduler.shift),
        }
        if int(representation_latents.shape[2]) > 1:
            delta = representation_latents[:, :, 1:] - representation_latents[:, :, :-1]
            metrics["repr/delta_norm"] = self._safe_scalar(delta.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt())
        if self.representation_prediction_type == "x":
            metrics["repr/x_mse"] = self._safe_scalar(F.mse_loss(pred_repr.float(), clean_future.float(), reduction="none"))
        if self.representation_prediction_type == "x" and self.representation_x_loss_mode == "velocity_equivalent":
            sigma = self._sigma_from_timestep(timestep_repr, pred_repr).clamp_min(self.representation_t_eps)
            v_pred = (noisy_future - pred_repr) / sigma
            v_target = noise_future - clean_future
            metrics["repr/v_equiv_mse"] = self._safe_scalar(F.mse_loss(v_pred.float(), v_target.float(), reduction="none"))
        else:
            del target_repr
        return metrics

    def training_loss(self, sample, tiled: bool = False):
        del tiled
        inputs = self.build_inputs(sample)
        representation_latents = inputs["representation_latents"]
        batch_size = int(representation_latents.shape[0])
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_repr = torch.randn_like(representation_latents)
        timestep_repr = self.train_representation_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=representation_latents.dtype,
        )
        noisy_repr = self.train_representation_scheduler.add_noise(representation_latents, noise_repr, timestep_repr)
        target_repr = self._representation_training_target(
            representation_latents=representation_latents,
            noise_repr=noise_repr,
            noisy_repr=noisy_repr,
            timestep_repr=timestep_repr,
        )
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
            action=None,
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
        clean_future = representation_latents[:, :, 1:]
        noisy_future = noisy_repr[:, :, 1:]
        noise_future = noise_repr[:, :, 1:]
        self._maybe_store_representation_visualization(
            pred_repr=pred_repr,
            target_repr=clean_future if self.representation_prediction_type == "x" else target_repr,
        )
        loss_repr_per_sample = self._compute_representation_prediction_loss_per_sample(
            pred_repr=pred_repr,
            target_repr=target_repr,
            clean_repr=clean_future,
            noisy_repr=noisy_future,
            noise_repr=noise_future,
            timestep_repr=timestep_repr,
            image_is_pad=image_is_pad,
        )
        repr_weight = self.train_representation_scheduler.training_weight(timestep_repr).to(
            loss_repr_per_sample.device, dtype=loss_repr_per_sample.dtype
        )
        loss_repr_raw = loss_repr_per_sample.mean()
        loss_repr = (loss_repr_per_sample * repr_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_representation * loss_repr + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_representation": self.loss_lambda_representation * float(loss_repr.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        loss_dict.update(
            self._representation_monitor_metrics(
                representation_latents=representation_latents,
                clean_future=clean_future,
                noisy_future=noisy_future,
                noise_future=noise_future,
                pred_repr=pred_repr,
                target_repr=target_repr,
                timestep_repr=timestep_repr,
                loss_repr_raw=loss_repr_raw,
                loss_repr=loss_repr,
            )
        )
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
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
        first_frame_repr = self._encode_inference_first_frame_latents(input_image)
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

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "model_class": self.__class__.__name__,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
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
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


class RARAE(RA):
    """RA with RAEv2-style representation-latent training defaults."""

    def __init__(self, *args, representation_prediction: Optional[dict[str, Any]] = None, **kwargs):
        if representation_prediction is None:
            representation_prediction = {"type": "x", "x_loss_mode": "direct", "t_eps": 0.05}
        super().__init__(*args, representation_prediction=representation_prediction, **kwargs)
