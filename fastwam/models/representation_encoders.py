from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


def _as_hw(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"Expected 2D size, got {value}.")
    return int(value[0]), int(value[1])


class BaseRepresentationEncoder(nn.Module):
    """Frozen visual teacher wrapper returning dense features as [B, C, T, H, W]."""

    output_dim: int

    def __init__(
        self,
        *,
        input_size: tuple[int, int],
        output_dim: int,
        camera_layout: str = "none",
        num_cameras: int = 1,
    ):
        super().__init__()
        self.input_size = _as_hw(input_size, input_size)
        self.output_dim = int(output_dim)
        self.camera_layout = str(camera_layout).lower()
        self.num_cameras = int(num_cameras)
        self._target_dtype: Optional[torch.dtype] = None

    @property
    def model(self) -> Optional[nn.Module]:
        return self.__dict__.get("_teacher_model", None)

    def _set_teacher_model(self, model: nn.Module) -> nn.Module:
        object.__setattr__(self, "_teacher_model", model)
        return model

    def to(self, *args, **kwargs):
        self._record_target_dtype(*args, **kwargs)
        super().to(*args, **kwargs)
        if self.model is not None:
            self.model.to(*args, **kwargs)
        return self

    def _record_target_dtype(self, *args, **kwargs):
        dtype = kwargs.get("dtype", None)
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
                if isinstance(arg, torch.Tensor):
                    dtype = arg.dtype
                    break
        if isinstance(dtype, torch.dtype):
            self._target_dtype = dtype

    def _floating_target_dtype(self) -> Optional[torch.dtype]:
        dtype = self._target_dtype
        if dtype is None:
            return None
        if torch.empty((), dtype=dtype).is_floating_point():
            return dtype
        return None

    def _model_to_kwargs(self, device: torch.device) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"device": device}
        dtype = self._floating_target_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype
        return kwargs

    def _cast_teacher_input(self, tensor: torch.Tensor) -> torch.Tensor:
        dtype = self._floating_target_dtype()
        if dtype is None or tensor.dtype == dtype:
            return tensor
        return tensor.to(dtype=dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.model is not None:
            self.model.eval()
        return self

    @property
    def feature_format(self) -> str:
        return "dense_bcthw"

    def split_cameras(self, video: torch.Tensor) -> list[torch.Tensor]:
        if self.camera_layout in {"none", "single", "null"}:
            return [video]
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,C,T,H,W], got {tuple(video.shape)}")
        _, _, _, height, width = video.shape
        if self.camera_layout == "robotwin":
            top_h = int(round(height * 2.0 / 3.0))
            if top_h <= 0 or top_h >= height:
                raise ValueError(f"Invalid robotwin split for H={height}.")
            half_w = width // 2
            if half_w * 2 != width:
                raise ValueError(f"RobotWin camera split requires even W, got {width}.")
            return [
                video[..., :top_h, :],
                video[..., top_h:, :half_w],
                video[..., top_h:, half_w:],
            ]
        if self.camera_layout == "horizontal":
            if width % self.num_cameras != 0:
                raise ValueError(f"Horizontal split requires W divisible by {self.num_cameras}, got {width}.")
            step = width // self.num_cameras
            return [video[..., i * step : (i + 1) * step] for i in range(self.num_cameras)]
        if self.camera_layout == "vertical":
            if height % self.num_cameras != 0:
                raise ValueError(f"Vertical split requires H divisible by {self.num_cameras}, got {height}.")
            step = height // self.num_cameras
            return [video[..., i * step : (i + 1) * step, :] for i in range(self.num_cameras)]
        raise ValueError(f"Unsupported camera_layout: {self.camera_layout}")

    def _video_to_images(self, video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = (video.detach().float() + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        batch_size, _, num_frames, _, _ = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x, batch_size, num_frames

    def _normalize_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (images - mean) / std

    def _tokens_to_dense(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        feat_h: int,
        feat_w: int,
        teacher_name: str,
    ) -> torch.Tensor:
        spatial_tokens = int(feat_h) * int(feat_w)
        if spatial_tokens <= 0:
            raise ValueError(f"Invalid dense grid for {teacher_name}: {(feat_h, feat_w)}")

        if tokens.ndim == 3:
            expected_batch = int(batch_size) * int(num_frames)
            if tokens.shape[0] != expected_batch:
                raise ValueError(
                    f"{teacher_name} token batch mismatch: got {tuple(tokens.shape)}, "
                    f"expected first dim {expected_batch}."
                )
            if tokens.shape[1] > spatial_tokens:
                tokens = tokens[:, -spatial_tokens:, :]
            if tokens.shape[1] != spatial_tokens:
                raise ValueError(
                    f"{teacher_name} spatial token count mismatch: "
                    f"got {tokens.shape[1]}, expected {spatial_tokens} for grid {(feat_h, feat_w)}."
                )
            return rearrange(
                tokens,
                "(b t) (h w) c -> b c t h w",
                b=int(batch_size),
                t=int(num_frames),
                h=int(feat_h),
                w=int(feat_w),
            ).contiguous()

        if tokens.ndim == 4:
            if tokens.shape[0] != int(batch_size) or tokens.shape[1] != int(num_frames):
                raise ValueError(
                    f"{teacher_name} token shape mismatch: got {tuple(tokens.shape)}, "
                    f"expected [B,T,N,C] with B={batch_size}, T={num_frames}."
                )
            if tokens.shape[2] > spatial_tokens:
                tokens = tokens[:, :, -spatial_tokens:, :]
            if tokens.shape[2] != spatial_tokens:
                raise ValueError(
                    f"{teacher_name} spatial token count mismatch: "
                    f"got {tokens.shape[2]}, expected {spatial_tokens} for grid {(feat_h, feat_w)}."
                )
            return rearrange(tokens, "b t (h w) c -> b c t h w", h=int(feat_h), w=int(feat_w)).contiguous()

        raise ValueError(f"{teacher_name} tokens must be [B*T,N,C] or [B,T,N,C], got {tuple(tokens.shape)}")

    def _check_camera_feature(self, feature: torch.Tensor, camera_idx: int) -> torch.Tensor:
        if feature.ndim != 5:
            raise ValueError(
                f"Camera feature {camera_idx} must be dense [B,C,T,H,W], got {tuple(feature.shape)}"
            )
        if feature.shape[1] != self.output_dim:
            raise ValueError(
                f"Camera feature {camera_idx} channel dim mismatch: "
                f"got {feature.shape[1]}, expected {self.output_dim}."
            )
        return feature

    def merge_camera_features(self, camera_features: list[torch.Tensor]) -> torch.Tensor:
        if not camera_features:
            raise ValueError("No camera features to merge.")
        features = [self._check_camera_feature(feature, idx) for idx, feature in enumerate(camera_features)]
        ref_b, ref_c, ref_t = features[0].shape[:3]
        for idx, feature in enumerate(features[1:], start=1):
            if feature.shape[:3] != (ref_b, ref_c, ref_t):
                raise ValueError(
                    "Camera features must share batch/channel/time dims: "
                    f"camera0={tuple(features[0].shape)} camera{idx}={tuple(feature.shape)}."
                )

        if self.camera_layout in {"none", "single", "null"}:
            if len(features) != 1:
                raise ValueError(f"Single-camera layout expected 1 feature map, got {len(features)}.")
            return features[0].contiguous()

        if self.camera_layout == "robotwin":
            if len(features) != 3:
                raise ValueError(f"RobotWin feature merge expects exactly 3 cameras, got {len(features)}.")
            top = features[0]
            top_h, top_w = int(top.shape[-2]), int(top.shape[-1])
            wrist_h = max(top_h // 2, 1)
            left_w = max(top_w // 2, 1)
            right_w = max(top_w - left_w, 1)
            left = F.interpolate(
                rearrange(features[1], "b c t h w -> (b t) c h w"),
                size=(wrist_h, left_w),
                mode="bilinear",
                align_corners=False,
            )
            right = F.interpolate(
                rearrange(features[2], "b c t h w -> (b t) c h w"),
                size=(wrist_h, right_w),
                mode="bilinear",
                align_corners=False,
            )
            left = rearrange(left, "(b t) c h w -> b c t h w", b=ref_b, t=ref_t)
            right = rearrange(right, "(b t) c h w -> b c t h w", b=ref_b, t=ref_t)
            return torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2).contiguous()

        if self.camera_layout == "horizontal":
            if len(features) != self.num_cameras:
                raise ValueError(f"Horizontal feature merge expected {self.num_cameras} cameras, got {len(features)}.")
            return torch.cat(features, dim=-1).contiguous()

        if self.camera_layout == "vertical":
            if len(features) != self.num_cameras:
                raise ValueError(f"Vertical feature merge expected {self.num_cameras} cameras, got {len(features)}.")
            return torch.cat(features, dim=-2).contiguous()

        raise ValueError(f"Unsupported camera_layout: {self.camera_layout}")

    def _freeze_loaded_model(self, model: nn.Module) -> nn.Module:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    @torch.inference_mode()
    def forward_pixels(self, video: torch.Tensor) -> torch.Tensor:
        camera_features = []
        for camera_video in self.split_cameras(video):
            camera_features.append(self.forward_camera_dense(camera_video))
        return self.merge_camera_features(camera_features)

    @torch.inference_mode()
    def forward_camera_dense(self, video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VGGTRepresentationEncoder(BaseRepresentationEncoder):
    def __init__(
        self,
        model_id: str = "facebook/VGGT-Omega",
        model_path: Optional[str] = None,
        input_size: tuple[int, int] = (512, 512),
        patch_size: int = 16,
        layer_index: int = -1,
        output_dim: int = 2048,
        camera_layout: str = "none",
        num_cameras: int = 1,
        repo_path: Optional[str] = "fastwam/third_party/vggt-omega",
        checkpoint_strict: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            output_dim=output_dim,
            camera_layout=camera_layout,
            num_cameras=num_cameras,
        )
        self.model_id = model_id
        self.model_path = model_path
        self.patch_size = int(patch_size)
        self.layer_index = int(layer_index)
        self.repo_path = repo_path
        self.checkpoint_strict = bool(checkpoint_strict)

    def _insert_repo_path(self):
        import sys

        candidates = []
        if self.repo_path not in (None, ""):
            candidates.append(str(self.repo_path))
        candidates.extend(["fastwam/third_party/vggt-omega", "third_party/vggt-omega"])
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists():
                resolved = str(path.resolve())
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)
                return

    def _resolve_checkpoint_path(self) -> Path:
        candidates = []
        for value in (self.model_path, self.model_id):
            if value not in (None, "", "null"):
                candidates.append(Path(str(value)).expanduser())

        model_leaf = str(self.model_id).rstrip("/").split("/")[-1]
        for root in ("checkpoints", "fastwam/checkpoints"):
            candidates.extend(
                [
                    Path(root) / str(self.model_id) / "vggt_omega_1b_512.pt",
                    Path(root) / str(self.model_id) / "model.pt",
                    Path(root) / model_leaf / "vggt_omega_1b_512.pt",
                    Path(root) / model_leaf / "model.pt",
                ]
            )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "VGGT-Omega requires a local checkpoint. Set `encoder.model_path` to "
            "`vggt_omega_1b_512.pt`, or place it under checkpoints/facebook/VGGT-Omega/."
        )

    def _checkpoint_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected VGGT-Omega checkpoint type at {checkpoint_path}: {type(payload)}")

        state_dict = {}
        for key, value in payload.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = str(key)
            for prefix in ("module.", "model."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            state_dict[clean_key] = value
        return state_dict

    def _load(self, device: torch.device):
        if self.model is not None:
            return
        self._insert_repo_path()
        try:
            from vggt_omega.models import VGGTOmega
        except Exception as exc:
            raise ImportError(
                "VGGT-Omega online extraction requires `vggt_omega` to be installed, or "
                "`encoder.repo_path` to point to the VGGT-Omega repo."
            ) from exc

        checkpoint_path = self._resolve_checkpoint_path()
        model = VGGTOmega(patch_size=self.patch_size, enable_camera=False, enable_depth=False, enable_alignment=False)
        load_result = model.load_state_dict(
            self._checkpoint_state_dict(checkpoint_path),
            strict=self.checkpoint_strict,
        )
        missing_aggregator = [key for key in load_result.missing_keys if key.startswith("aggregator.")]
        if missing_aggregator:
            raise ValueError(
                f"VGGT-Omega checkpoint is missing aggregator weights, e.g. {missing_aggregator[:5]}."
            )
        self._set_teacher_model(
            self._freeze_loaded_model(model.to(**self._model_to_kwargs(device)))
        )

    @torch.inference_mode()
    def forward_pixels(self, video: torch.Tensor) -> torch.Tensor:
        self._load(video.device)
        camera_videos = self.split_cameras(video)
        resized_views = []
        batch_size = None
        num_frames = None
        for camera_video in camera_videos:
            images, camera_batch_size, camera_num_frames = self._video_to_images(camera_video)
            batch_size = camera_batch_size if batch_size is None else batch_size
            num_frames = camera_num_frames if num_frames is None else num_frames
            if camera_batch_size != batch_size or camera_num_frames != num_frames:
                raise ValueError("VGGT camera views must share batch/time dimensions.")
            resized_views.append(rearrange(images, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames))

        views = torch.stack(resized_views, dim=2)
        views = rearrange(views, "b t v c h w -> b (t v) c h w")
        views = self._cast_teacher_input(views)
        tokens = self._forward_tokens(views)
        if tokens.ndim != 4:
            raise ValueError(f"Expected VGGT-Omega tokens [B,T*V,N,C], got {tuple(tokens.shape)}")
        if tokens.shape[0] != int(batch_size) or tokens.shape[1] != int(num_frames) * len(camera_videos):
            raise ValueError(
                "VGGT-Omega token shape mismatch: "
                f"got {tuple(tokens.shape)}, expected first dims {(int(batch_size), int(num_frames) * len(camera_videos))}."
            )

        feat_h = self.input_size[0] // self.patch_size
        feat_w = self.input_size[1] // self.patch_size
        if tokens.shape[2] > feat_h * feat_w:
            tokens = tokens[:, :, -feat_h * feat_w :, :]
        if tokens.shape[2] != feat_h * feat_w:
            raise ValueError(f"VGGT-Omega token count mismatch: got {tokens.shape[2]}, expected {feat_h * feat_w}.")

        maps = rearrange(
            tokens,
            "b (t v) (h w) c -> b t v c h w",
            t=int(num_frames),
            v=len(camera_videos),
            h=feat_h,
            w=feat_w,
        )
        camera_features = [
            rearrange(maps[:, :, view], "b t c h w -> b c t h w").contiguous()
            for view in range(maps.shape[2])
        ]
        return self.merge_camera_features(camera_features)

    def _forward_tokens(self, views: torch.Tensor) -> torch.Tensor:
        tokens_list, patch_token_start = self.model.aggregator(views)
        tokens = tokens_list[self.layer_index]
        if tokens is None:
            available = [idx for idx, item in enumerate(tokens_list) if item is not None]
            raise ValueError(
                f"VGGT-Omega layer_index={self.layer_index} was not cached. "
                f"Available cached layer indices: {available}"
            )
        return tokens[:, :, patch_token_start:].contiguous()

    @torch.inference_mode()
    def forward_camera_dense(self, video: torch.Tensor) -> torch.Tensor:
        self._load(video.device)
        images, batch_size, num_frames = self._video_to_images(video)
        images = self._cast_teacher_input(images)
        views = rearrange(images, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames)
        tokens = self._forward_tokens(views)
        if tokens.ndim != 4:
            raise ValueError(f"Unexpected VGGT-Omega token shape: {tuple(tokens.shape)}")
        h = self.input_size[0] // self.patch_size
        w = self.input_size[1] // self.patch_size
        return self._tokens_to_dense(
            tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            feat_h=h,
            feat_w=w,
            teacher_name="VGGT-Omega",
        )


class DINORepresentationEncoder(BaseRepresentationEncoder):
    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        backend: str = "hub",
        hub_repo: str = "facebookresearch/dinov3",
        hub_source: str = "github",
        hub_kwargs: Optional[dict[str, Any]] = None,
        input_size: tuple[int, int] = (512, 512),
        patch_size: int = 16,
        output_dim: int = 768,
        camera_layout: str = "none",
        num_cameras: int = 1,
    ):
        super().__init__(
            input_size=input_size,
            output_dim=output_dim,
            camera_layout=camera_layout,
            num_cameras=num_cameras,
        )
        self.model_name = model_name
        self.model_id = model_id
        self.backend = str(backend).lower()
        self.hub_repo = hub_repo
        self.hub_source = hub_source
        self.hub_kwargs = _as_plain_dict(hub_kwargs)
        self.patch_size = int(patch_size)

    def _load(self, device: torch.device):
        if self.model is not None:
            return
        if self.backend == "hub":
            model = torch.hub.load(
                self.hub_repo,
                self.model_name,
                source=self.hub_source,
                **self.hub_kwargs,
            )
        elif self.backend in {"hf", "transformers"}:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported DINO backend: {self.backend}")
        self._set_teacher_model(self._freeze_loaded_model(model.to(device)))

    @torch.no_grad()
    def forward_camera_dense(self, video: torch.Tensor) -> torch.Tensor:
        self._load(video.device)
        images, batch_size, num_frames = self._video_to_images(video)
        images = self._normalize_imagenet(images)
        feat_h = self.input_size[0] // self.patch_size
        feat_w = self.input_size[1] // self.patch_size
        spatial_tokens = feat_h * feat_w
        if self.backend == "hub":
            out = self.model.forward_features(images) if hasattr(self.model, "forward_features") else self.model(images)
            if isinstance(out, dict):
                tokens = out.get("x_norm_patchtokens", out.get("patch_tokens", None))
                if tokens is None:
                    raise ValueError(f"DINO forward_features output has no patch token key: {sorted(out.keys())}")
            else:
                tokens = out
                if tokens.ndim == 3 and tokens.shape[1] == spatial_tokens + 1:
                    tokens = tokens[:, 1:, :]
        else:
            out = self.model(pixel_values=images)
            tokens = getattr(out, "last_hidden_state", None)
            if tokens is None:
                raise ValueError("DINO output does not expose `last_hidden_state`.")
            if tokens.shape[1] == spatial_tokens + 1:
                tokens = tokens[:, 1:, :]
        return self._tokens_to_dense(
            tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            feat_h=feat_h,
            feat_w=feat_w,
            teacher_name="DINO",
        )


class DINOMultiLayerSumRepresentationEncoder(DINORepresentationEncoder):
    """DINO/DINOv3 dense features from a sum of normalized intermediate layers."""

    def __init__(
        self,
        *args,
        layers: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if layers in (None, [], ""):
            layers = [11, 13, 15, 17, 19, 21, 23]
        self.layers = [int(layer) for layer in layers]
        if not self.layers:
            raise ValueError("DINO multi-layer-sum requires at least one layer index.")

    @torch.no_grad()
    def forward_camera_dense(self, video: torch.Tensor) -> torch.Tensor:
        self._load(video.device)
        images, batch_size, num_frames = self._video_to_images(video)
        images = self._normalize_imagenet(images)
        feat_h = self.input_size[0] // self.patch_size
        feat_w = self.input_size[1] // self.patch_size
        spatial_tokens = feat_h * feat_w

        if self.backend != "hub":
            raise ValueError("DINO multi-layer-sum currently requires backend='hub' with get_intermediate_layers.")
        if not hasattr(self.model, "get_intermediate_layers"):
            raise ValueError("DINO model does not expose `get_intermediate_layers`, required for multi-layer-sum.")

        outputs = self.model.get_intermediate_layers(
            images,
            n=self.layers,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise ValueError("DINO get_intermediate_layers returned no tensor outputs.")
        layer_tokens = []
        for output in outputs:
            if isinstance(output, (list, tuple)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"Unexpected DINO intermediate output type: {type(output)!r}.")
            if output.ndim != 3:
                raise ValueError(f"DINO intermediate output must be [B,N,C], got {tuple(output.shape)}.")
            if output.shape[1] > spatial_tokens:
                output = output[:, -spatial_tokens:, :]
            layer_tokens.append(output)
        tokens = torch.stack(layer_tokens, dim=0).sum(dim=0)
        return self._tokens_to_dense(
            tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            feat_h=feat_h,
            feat_w=feat_w,
            teacher_name="DINO-MLS",
        )


class VJEPARepresentationEncoder(BaseRepresentationEncoder):
    def __init__(
        self,
        model_id: str = "",
        backend: str = "hub",
        hub_repo: str = "facebookresearch/vjepa2",
        hub_model_name: str = "vjepa2_1_vit_large_384",
        hub_source: str = "github",
        hub_kwargs: Optional[dict[str, Any]] = None,
        input_layout: str = "bcthw",
        input_size: tuple[int, int] = (384, 384),
        output_dim: int = 1024,
        patch_size: Optional[int] = 16,
        camera_layout: str = "none",
        num_cameras: int = 1,
    ):
        super().__init__(
            input_size=input_size,
            output_dim=output_dim,
            camera_layout=camera_layout,
            num_cameras=num_cameras,
        )
        self.model_id = model_id
        self.backend = str(backend).lower()
        self.hub_repo = hub_repo
        self.hub_model_name = hub_model_name
        self.hub_source = hub_source
        self.hub_kwargs = _as_plain_dict(hub_kwargs)
        self.input_layout = str(input_layout).lower()
        self.patch_size = None if patch_size is None else int(patch_size)

    def _load(self, device: torch.device):
        if self.model is not None:
            return
        if self.backend in {"hf", "transformers"}:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        elif self.backend in {"hub", "torchhub"}:
            model = torch.hub.load(
                self.hub_repo,
                self.hub_model_name,
                source=self.hub_source,
                **self.hub_kwargs,
            )
            if isinstance(model, (list, tuple)):
                if not model:
                    raise ValueError(f"V-JEPA hub model `{self.hub_model_name}` returned an empty tuple/list.")
                model = model[0]
        else:
            raise ValueError(f"Unsupported V-JEPA backend: {self.backend}")
        self._set_teacher_model(self._freeze_loaded_model(model.to(device)))

    def _extract_tokens(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
            raise ValueError("V-JEPA output tuple/list has no tensor payload.")
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None:
            tokens = getattr(output, "encoder_last_hidden_state", None)
        if tokens is None:
            predictor_output = getattr(output, "predictor_output", None)
            tokens = getattr(predictor_output, "last_hidden_state", None)
        if tokens is None:
            raise ValueError("V-JEPA output does not expose dense hidden states.")
        return tokens

    @torch.no_grad()
    def forward_camera_dense(self, video: torch.Tensor) -> torch.Tensor:
        self._load(video.device)
        x = (video.detach().float() + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        batch_size, _, num_frames, _, _ = x.shape
        frames_per_clip = int(num_frames)
        x = F.interpolate(
            x,
            size=(frames_per_clip, self.input_size[0], self.input_size[1]),
            mode="trilinear",
            align_corners=False,
        )
        x = rearrange(x, "b c t h w -> b t c h w")
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        x = (x - mean) / std

        if self.backend in {"hf", "transformers"}:
            out = self.model(pixel_values_videos=x, skip_predictor=True)
        else:
            if self.input_layout == "btchw":
                model_input = x
            elif self.input_layout == "bcthw":
                model_input = rearrange(x, "b t c h w -> b c t h w")
            else:
                raise ValueError(f"Unsupported V-JEPA input_layout: {self.input_layout}")
            out = self.model(model_input)
        tokens = self._extract_tokens(out)

        if tokens.ndim == 4:
            if tokens.shape[0] != batch_size:
                raise ValueError(f"Unexpected V-JEPA token batch shape: {tuple(tokens.shape)}")
            patch_size = self.patch_size
            if patch_size is None:
                patch_size = int(getattr(self.model.config, "patch_size", 16))
            feat_h = self.input_size[0] // patch_size
            feat_w = self.input_size[1] // patch_size
            return self._tokens_to_dense(
                tokens,
                batch_size=batch_size,
                num_frames=int(tokens.shape[1]),
                feat_h=feat_h,
                feat_w=feat_w,
                teacher_name="V-JEPA",
            )
        if tokens.ndim != 3 or tokens.shape[0] != batch_size:
            raise ValueError(f"Unexpected V-JEPA token shape: {tuple(tokens.shape)}")

        patch_size = self.patch_size
        if patch_size is None:
            patch_size = int(getattr(self.model.config, "patch_size", 16))
        spatial_tokens = (self.input_size[0] // patch_size) * (self.input_size[1] // patch_size)
        if spatial_tokens <= 0:
            raise ValueError(f"Invalid V-JEPA spatial token count for input_size={self.input_size}, patch_size={patch_size}.")
        if tokens.shape[1] > 1:
            without_cls = tokens[:, 1:, :]
            if without_cls.shape[1] % spatial_tokens == 0:
                tokens = without_cls
        if tokens.shape[1] % spatial_tokens != 0:
            raise ValueError(
                "Cannot reshape V-JEPA tokens into frame tokens: "
                f"num_tokens={tokens.shape[1]}, spatial_tokens={spatial_tokens}."
            )
        temporal_tokens = tokens.shape[1] // spatial_tokens
        tokens = rearrange(tokens, "b (t n) c -> b t n c", t=temporal_tokens, n=spatial_tokens)
        feat_h = self.input_size[0] // patch_size
        feat_w = self.input_size[1] // patch_size
        return self._tokens_to_dense(
            tokens,
            batch_size=batch_size,
            num_frames=temporal_tokens,
            feat_h=feat_h,
            feat_w=feat_w,
            teacher_name="V-JEPA",
        )


def build_representation_encoder(cfg: dict[str, Any]) -> BaseRepresentationEncoder:
    cfg = _as_plain_dict(cfg)
    encoder_cfg = _as_plain_dict(cfg.get("encoder", {}))
    name = str(
        encoder_cfg.pop("name", cfg.get("teacher", cfg.get("encoder_type", "vggt")))
    ).lower()
    if "feature_dim" in cfg and "output_dim" not in encoder_cfg:
        encoder_cfg["output_dim"] = int(cfg["feature_dim"])
    if name in {"vggt", "vggt_omega", "vggt-omega"}:
        return VGGTRepresentationEncoder(**encoder_cfg)
    if name == "dino":
        return DINORepresentationEncoder(**encoder_cfg)
    if name in {"dino_mls", "dinomls", "dinov3_mls", "dinov3mls"}:
        return DINOMultiLayerSumRepresentationEncoder(**encoder_cfg)
    if name == "vjepa":
        return VJEPARepresentationEncoder(**encoder_cfg)
    raise ValueError(f"Unsupported representation encoder: {name}")
