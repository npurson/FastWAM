from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastwam.models.representation_encoders import build_representation_encoder


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class RepresentationForcing(nn.Module):
    """Online VrfA targets and heads for REPA, Geometry Forcing, and ReDi."""

    def __init__(self, hidden_dim: int, cfg: Any):
        super().__init__()
        cfg = _as_plain_dict(cfg)
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "repa")).lower()
        self.hidden_dim = int(hidden_dim)
        self.encoder = build_representation_encoder(cfg) if self.enabled else None
        self.feature_dim = int(
            cfg.get(
                "feature_dim",
                getattr(self.encoder, "output_dim", cfg.get("target_dim", 2048)),
            )
        )

        pca_cfg = _as_plain_dict(cfg.get("pca", {}))
        self.pca_enabled = bool(pca_cfg.get("enabled", False))
        self.pca_dim = int(pca_cfg.get("dim", 16))
        self.pca_fit_on_the_fly = bool(pca_cfg.get("fit_on_the_fly", False))
        self.pca_stats_path = pca_cfg.get("stats_path", None)
        self.target_dim = int(cfg.get("target_dim", self.pca_dim if self.pca_enabled else self.feature_dim))
        if self.pca_enabled and self.target_dim != self.pca_dim:
            raise ValueError(
                f"When PCA is enabled, `target_dim` must equal `pca.dim`; got {self.target_dim} vs {self.pca_dim}."
            )

        self.loss_weight = float(cfg.get("loss_weight", 1.0))
        self.angular_weight = float(cfg.get("angular_weight", 1.0))
        self.scale_weight = float(cfg.get("scale_weight", 1.0))
        self.redi_drop_prob = float(cfg.get("redi_drop_prob", 0.0))
        self.redi_condition_first_frame = bool(cfg.get("redi_condition_first_frame", False))
        self.normalize_target = bool(cfg.get("normalize_target", False))
        self.detach_alignment_input = bool(cfg.get("detach_alignment_input", False))
        self.target_temporal_sampling = str(cfg.get("target_temporal_sampling", "all_interpolate")).lower()
        self.layer_indices = [int(v) for v in cfg.get("layer_indices", [15])]
        self.projector_hidden_dim = int(cfg.get("projector_hidden_dim", hidden_dim))

        self.register_buffer("pca_mean", torch.empty(0), persistent=False)
        self.register_buffer("pca_components", torch.empty(0), persistent=False)
        if self.pca_enabled and self.pca_stats_path not in (None, "", "null"):
            self._load_pca_stats(str(self.pca_stats_path))

        if not self.enabled:
            return
        if self.mode not in {"repa", "gf", "redi"}:
            raise ValueError(f"Unsupported representation forcing mode: {self.mode}")
        if self.target_temporal_sampling not in {"all_interpolate", "latent_anchor_center"}:
            raise ValueError(
                "Unsupported representation forcing target_temporal_sampling: "
                f"{self.target_temporal_sampling}. Expected one of: all_interpolate, latent_anchor_center."
            )

        if self.mode == "repa":
            self.repa_projectors = nn.ModuleDict(
                {
                    str(layer): _build_mlp(hidden_dim, self.projector_hidden_dim, self.target_dim)
                    for layer in self.layer_indices
                }
            )
        elif self.mode == "gf":
            self.gf_projectors = nn.ModuleDict(
                {
                    str(layer): _build_mlp(hidden_dim, self.projector_hidden_dim, self.target_dim)
                    for layer in self.layer_indices
                }
            )
            self.gf_unnormalizers = nn.ModuleDict(
                {
                    str(layer): _build_mlp(self.target_dim, self.projector_hidden_dim, self.target_dim)
                    for layer in self.layer_indices
                }
            )
        else:
            self.representation_embedding = nn.Linear(self.target_dim, hidden_dim)
            self.representation_head = _build_mlp(hidden_dim, self.projector_hidden_dim, self.target_dim)

    @property
    def requires_intermediate_layers(self) -> bool:
        return self.enabled and self.mode in {"repa", "gf"} and bool(self.layer_indices)

    def train_heads_only(self):
        self.train()
        self.requires_grad_(True)
        if self.encoder is not None:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            model = getattr(self.encoder, "model", None)
            if model is not None:
                model.eval()
                for param in model.parameters():
                    param.requires_grad_(False)
        return self

    def _load_pca_stats(self, path: str):
        payload = torch.load(path, map_location="cpu")
        if "mean" not in payload or "components" not in payload:
            raise ValueError(f"PCA stats must contain `mean` and `components`: {path}")
        mean = payload["mean"].float().flatten()
        components = payload["components"].float()
        if components.ndim != 2:
            raise ValueError(f"`components` must be 2D, got {tuple(components.shape)} in {path}")
        if components.shape[1] == self.feature_dim and components.shape[0] >= self.pca_dim:
            components = components[: self.pca_dim].contiguous()
        elif components.shape[0] == self.feature_dim and components.shape[1] >= self.pca_dim:
            components = components.t().contiguous()
            components = components[: self.pca_dim].contiguous()
        else:
            raise ValueError(
                f"PCA components must be [num_components>=dim, feature_dim] or [feature_dim, num_components>=dim], "
                f"got {tuple(components.shape)}, dim={self.pca_dim}, feature_dim={self.feature_dim}."
            )
        if mean.numel() != self.feature_dim:
            raise ValueError(f"PCA mean dim mismatch: got {mean.numel()}, expected {self.feature_dim}.")
        self.pca_mean = mean
        self.pca_components = components

    def _apply_pca(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.pca_enabled:
            return tokens
        tokens_float = tokens.float()
        if self.pca_components.numel() > 0:
            mean = self.pca_mean.to(device=tokens.device, dtype=tokens_float.dtype)
            components = self.pca_components.to(device=tokens.device, dtype=tokens_float.dtype)
            return (tokens_float - mean) @ components.t()
        if not self.pca_fit_on_the_fly:
            raise ValueError(
                "PCA is enabled but no PCA stats were loaded. Set "
                "`representation_forcing.pca.stats_path=/path/to/pca.pt`, or set "
                "`representation_forcing.pca.fit_on_the_fly=true` for a slow debug path."
            )
        flat = tokens_float.reshape(-1, tokens_float.shape[-1])
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        _, _, v = torch.pca_lowrank(centered, q=self.pca_dim, center=False)
        projected = centered @ v[:, : self.pca_dim]
        return projected.reshape(*tokens.shape[:-1], self.pca_dim)

    def _tokens_from_features(
        self,
        features: torch.Tensor,
        grid_size: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        f, h, w = [int(v) for v in grid_size]
        features = features.to(device=device, dtype=torch.float32)

        if features.ndim == 4 and features.shape[-1] == self.feature_dim:
            tokens = rearrange(features, "b t n c -> b c t n")
            tokens = F.interpolate(tokens, size=(f, h * w), mode="bilinear", align_corners=False)
            tokens = rearrange(tokens, "b c t n -> b (t n) c")
        elif features.ndim == 5 and features.shape[1] == self.feature_dim:
            features = F.interpolate(features, size=(f, h, w), mode="trilinear", align_corners=False)
            tokens = rearrange(features, "b c t h w -> b (t h w) c")
        else:
            raise ValueError(
                "Online representation encoders must return [B,T,N,C] tokens or [B,C,T,H,W] dense features; "
                f"got {tuple(features.shape)} with feature_dim={self.feature_dim}."
            )

        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Representation feature dim mismatch: got {tokens.shape[-1]}, expected {self.feature_dim}. "
                "Set `representation_forcing.feature_dim` or `representation_forcing.encoder.output_dim` correctly."
            )
        tokens = self._apply_pca(tokens)
        if tokens.shape[-1] != self.target_dim:
            raise ValueError(f"Target feature dim mismatch after PCA: got {tokens.shape[-1]}, expected {self.target_dim}.")
        if self.normalize_target:
            mean = tokens.mean(dim=(1, 2), keepdim=True)
            std = tokens.std(dim=(1, 2), keepdim=True)
            tokens = (tokens - mean) / (std + 1e-6)
        return tokens.to(dtype=dtype)

    def get_target_tokens(
        self,
        input_video: torch.Tensor,
        grid_size: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.enabled:
            raise ValueError("Representation forcing is disabled.")
        if self.encoder is None:
            raise ValueError("Online representation encoder was not initialized.")
        with torch.inference_mode():
            input_video = self._sample_target_video_frames(input_video, grid_size=grid_size)
            features = self.encoder.forward_pixels(input_video)
            tokens = self._tokens_from_features(features, grid_size=grid_size, device=device, dtype=dtype)
        return tokens.clone().detach()

    def _sample_target_video_frames(
        self,
        input_video: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        if self.target_temporal_sampling == "all_interpolate":
            return input_video
        if self.target_temporal_sampling != "latent_anchor_center":
            raise ValueError(f"Unsupported target_temporal_sampling: {self.target_temporal_sampling}")
        if input_video.ndim != 5:
            raise ValueError(f"`input_video` must be [B,C,T,H,W], got {tuple(input_video.shape)}")

        target_frames = int(grid_size[0])
        source_frames = int(input_video.shape[2])
        if target_frames <= 0:
            raise ValueError(f"Invalid target temporal grid size: {target_frames}")
        if source_frames <= 0:
            raise ValueError(f"Invalid input video temporal size: {source_frames}")
        if target_frames >= source_frames:
            return input_video
        if target_frames == 1:
            indices = [0]
        else:
            tail_frames = source_frames - 1
            tail_targets = target_frames - 1
            indices = [0]
            for target_idx in range(1, target_frames):
                start = 1 + (target_idx - 1) * tail_frames // tail_targets
                end = 1 + target_idx * tail_frames // tail_targets
                anchor = (start + end) // 2
                indices.append(min(max(anchor, 0), source_frames - 1))
        index_tensor = torch.as_tensor(indices, device=input_video.device, dtype=torch.long)
        return input_video.index_select(dim=2, index=index_tensor)

    def apply_redi_input(
        self,
        video_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        timestep: torch.Tensor,
        scheduler,
        video_tokens_per_frame: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if not self.enabled or self.mode != "redi":
            return video_tokens, None
        noise = torch.randn_like(target_tokens)
        noisy = scheduler.add_noise(target_tokens, noise, timestep)
        if self.training and self.redi_drop_prob > 0:
            drop = torch.rand((noisy.shape[0], 1, 1), device=noisy.device) < self.redi_drop_prob
            noisy = torch.where(drop, torch.zeros_like(noisy), noisy)
        repr_emb = self.representation_embedding(noisy)
        if not self.redi_condition_first_frame and video_tokens_per_frame is not None:
            repr_emb = repr_emb.clone()
            repr_emb[:, : int(video_tokens_per_frame)] = 0
        video_tokens = video_tokens + repr_emb
        target = scheduler.training_target(target_tokens, noise, timestep)
        return video_tokens, {"target": target}

    def apply_redi_inference_input(
        self,
        video_tokens: torch.Tensor,
        representation_latents: Optional[torch.Tensor],
        video_tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.mode != "redi":
            return video_tokens
        if representation_latents is None:
            raise ValueError("ReDi inference requires representation latents.")
        repr_emb = self.representation_embedding(representation_latents)
        if not self.redi_condition_first_frame and video_tokens_per_frame is not None:
            repr_emb = repr_emb.clone()
            repr_emb[:, : int(video_tokens_per_frame)] = 0
        return video_tokens + repr_emb

    def predict_redi_velocity(self, final_video_tokens: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mode != "redi":
            raise ValueError("`predict_redi_velocity` is only valid for mode='redi'.")
        return self.representation_head(final_video_tokens)

    @staticmethod
    def _resolve_alignment_hidden(
        *,
        layer: int,
        final_video_tokens: torch.Tensor,
        intermediate_video_tokens: Optional[dict[int, torch.Tensor]],
        mode_name: str,
    ) -> torch.Tensor:
        if intermediate_video_tokens is None:
            return final_video_tokens
        if layer not in intermediate_video_tokens:
            raise ValueError(
                f"{mode_name} forcing requested intermediate layer {layer}, "
                f"but available layers are {sorted(intermediate_video_tokens.keys())}."
            )
        return intermediate_video_tokens[layer]

    def visualization_tokens(
        self,
        *,
        target_tokens: torch.Tensor,
        final_video_tokens: torch.Tensor,
        intermediate_video_tokens: Optional[dict[int, torch.Tensor]] = None,
        redi_state: Optional[dict[str, torch.Tensor]] = None,
    ) -> Optional[tuple[str, torch.Tensor, torch.Tensor]]:
        """Return detached pred/target tokens for lightweight debug visualization."""
        if not self.enabled:
            return None

        if self.mode == "repa":
            layer = self.layer_indices[0]
            hidden = self._resolve_alignment_hidden(
                layer=layer,
                final_video_tokens=final_video_tokens,
                intermediate_video_tokens=intermediate_video_tokens,
                mode_name="REPA",
            )
            pred = self.repa_projectors[str(layer)](hidden)
            return f"repa_layer_{layer}", pred.detach(), target_tokens.detach()

        if self.mode == "gf":
            layer = self.layer_indices[0]
            hidden = self._resolve_alignment_hidden(
                layer=layer,
                final_video_tokens=final_video_tokens,
                intermediate_video_tokens=intermediate_video_tokens,
                mode_name="GF",
            )
            pred = self.gf_projectors[str(layer)](hidden)
            return f"gf_layer_{layer}", pred.detach(), target_tokens.detach()

        if self.mode == "redi":
            if redi_state is None or "target" not in redi_state:
                return None
            pred = self.representation_head(final_video_tokens)
            return "redi_velocity", pred.detach(), redi_state["target"].detach()

        return None

    def compute_loss(
        self,
        target_tokens: torch.Tensor,
        final_video_tokens: torch.Tensor,
        intermediate_video_tokens: Optional[dict[int, torch.Tensor]] = None,
        redi_state: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = final_video_tokens.new_zeros(())
        if not self.enabled:
            return zero, {}

        if self.mode == "repa":
            loss = zero
            count = 0
            for layer in self.layer_indices:
                hidden = self._resolve_alignment_hidden(
                    layer=layer,
                    final_video_tokens=final_video_tokens,
                    intermediate_video_tokens=intermediate_video_tokens,
                    mode_name="REPA",
                )
                if self.detach_alignment_input:
                    hidden = hidden.detach()
                pred = self.repa_projectors[str(layer)](hidden)
                loss = loss + (1.0 - F.cosine_similarity(pred.float(), target_tokens.float(), dim=-1)).mean()
                count += 1
            loss = loss / max(count, 1)
            return self.loss_weight * loss, {"loss_repr_repa": float(loss.detach().item())}

        if self.mode == "gf":
            angular = zero
            scale = zero
            count = 0
            target_float = target_tokens.float()
            for layer in self.layer_indices:
                hidden = self._resolve_alignment_hidden(
                    layer=layer,
                    final_video_tokens=final_video_tokens,
                    intermediate_video_tokens=intermediate_video_tokens,
                    mode_name="GF",
                )
                if self.detach_alignment_input:
                    hidden = hidden.detach()
                pred = self.gf_projectors[str(layer)](hidden).float()
                angular = angular + (1.0 - F.cosine_similarity(pred, target_float, dim=-1)).mean()
                pred_normalized = F.normalize(pred, p=2, dim=-1)
                pred_unnormalized = self.gf_unnormalizers[str(layer)](pred_normalized).float()
                scale = scale + F.mse_loss(pred_unnormalized, target_float, reduction="mean")
                count += 1
            angular = angular / max(count, 1)
            scale = scale / max(count, 1)
            loss = self.angular_weight * angular + self.scale_weight * scale
            return self.loss_weight * loss, {
                "loss_repr_angular": float(angular.detach().item()),
                "loss_repr_scale": float(scale.detach().item()),
            }

        if self.mode == "redi":
            if redi_state is None or "target" not in redi_state:
                raise ValueError("ReDi forcing requires `redi_state` from `apply_redi_input`.")
            pred = self.representation_head(final_video_tokens)
            loss = F.mse_loss(pred.float(), redi_state["target"].float(), reduction="mean")
            return self.loss_weight * loss, {"loss_repr_redi": float(loss.detach().item())}

        raise ValueError(f"Unsupported representation forcing mode: {self.mode}")
