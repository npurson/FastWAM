from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_3tuple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in value)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"`{name}` must contain three positive integers, got {values}.")
    return values


def _random_codec_weight(
    *,
    input_dim: int,
    output_dim: int,
    kernel_size: tuple[int, int, int],
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weight = torch.randn(
        int(output_dim),
        int(input_dim),
        *kernel_size,
        generator=generator,
        dtype=torch.float32,
    )
    return F.normalize(weight.flatten(1), dim=1).reshape_as(weight)


def _validate_temporal_padding(value: str) -> str:
    value = str(value).lower()
    if value not in {"left_zero", "none"}:
        raise ValueError(
            "representation.codec.temporal_padding must be one of "
            f"{'left_zero', 'none'}, got {value!r}."
        )
    return value


class FrozenRandomCausalCodec(nn.Module):
    """Frozen random linear codec with FastWAM-style first-frame causality.

    The temporal left pad makes an odd-length video follow this grouping:

        [frame0], [frame1, frame2], [frame3, frame4], ...

    Consequently, encoding a single image at inference produces the same first
    latent as encoding the first frame of a training clip. Spatial dimensions
    are reduced by the same strided projection.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        kernel_size: Sequence[int] = (2, 2, 2),
        stride: Sequence[int] = (2, 2, 2),
        seed: int = 0,
        temporal_padding: str = "left_zero",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_size = _as_3tuple(kernel_size, "representation.codec.kernel_size")
        self.stride = _as_3tuple(stride, "representation.codec.stride")
        self.seed = int(seed)
        self.temporal_padding = _validate_temporal_padding(temporal_padding)

        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError(
                "representation codec dimensions must be positive, "
                f"got input_dim={self.input_dim}, output_dim={self.output_dim}."
            )
        if self.kernel_size != (2, 2, 2) or self.stride != (2, 2, 2):
            raise ValueError(
                "FrozenRandomCausalCodec currently requires 2x2x2 kernel/stride, "
                f"got kernel={self.kernel_size}, stride={self.stride}."
            )

        weight = _random_codec_weight(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            kernel_size=self.kernel_size,
            seed=self.seed,
        )
        # Unit-norm rows preserve unit input variance without changing the
        # process-global RNG state. The projection remains frozen as a buffer.
        self.register_buffer("weight", weight, persistent=True)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, seed={self.seed}, "
            f"temporal_padding={self.temporal_padding!r}"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "FrozenRandomCausalCodec expects [B,C,T,H,W], "
                f"got {tuple(features.shape)}."
            )
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                "FrozenRandomCausalCodec channel mismatch: "
                f"got {features.shape[1]}, expected {self.input_dim}."
            )
        expected_remainder = 1 if self.temporal_padding == "left_zero" else 0
        if int(features.shape[2]) % 2 != expected_remainder:
            raise ValueError(
                "FrozenRandomCausalCodec temporal input mismatch: "
                f"temporal_padding={self.temporal_padding!r} requires "
                f"T % 2 == {expected_remainder}, got T={features.shape[2]}."
            )
        if int(features.shape[3]) % 2 != 0 or int(features.shape[4]) % 2 != 0:
            raise ValueError(
                "FrozenRandomCausalCodec requires even spatial dimensions, "
                f"got HxW={features.shape[3]}x{features.shape[4]}."
            )

        if self.temporal_padding == "left_zero":
            # Conv3d pads are ordered W-left/right, H-left/right, T-left/right.
            features = F.pad(features, (0, 0, 0, 0, 1, 0))
        latents = F.conv3d(
            features,
            self.weight.to(device=features.device, dtype=features.dtype),
            bias=None,
            stride=self.stride,
        )
        # Keep the tokenizer output raw. Diffusion-facing normalization belongs
        # after the codec so its statistics describe the latent that is actually
        # noised, while the detached decoder can continue to consume raw latents.
        return latents.contiguous()

    def expected_output_shape(self, input_shape: Sequence[int]) -> tuple[int, int, int, int, int]:
        if len(input_shape) != 5:
            raise ValueError(f"Expected a 5D input shape, got {tuple(input_shape)}.")
        batch, channels, frames, height, width = [int(v) for v in input_shape]
        if channels != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got C={channels}.")
        output_frames = math.ceil(frames / 2) if self.temporal_padding == "left_zero" else frames // 2
        return batch, self.output_dim, output_frames, height // 2, width // 2


class LearnableRandomCausalCodec(FrozenRandomCausalCodec):
    """Trainable codec initialized exactly like ``FrozenRandomCausalCodec``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        weight = self.weight.detach().clone()
        del self._buffers["weight"]
        self.register_parameter("weight", nn.Parameter(weight))


class FAEAttentionCausalCodec(nn.Module):
    """Single-attention codec over one paired multi-camera observation at a time.

    Each non-overlapping 2x2x2 feature block is first patchified without losing
    information. For the RobotWin layout, the resulting front, left-wrist, and
    right-wrist tokens from the same temporal pair are concatenated into one
    sequence. Five temporal pairs are still processed independently, preventing
    future-to-current leakage while allowing cross-camera correspondence.

    Q/K inspect the full patchified feature, while the value projection maps the
    8192-d production input directly to the diffusion-facing channel dimension.
    This is equivalent to applying the final linear projection after token
    mixing, but avoids carrying an 8192-d value path through attention. The
    value projection uses the exact frozen-random Conv3d initialization and the
    attention output projection starts at zero, so the initial output is exactly
    the existing linear codec.
    """

    num_cameras = 3

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        qk_dim: int = 1024,
        num_heads: int = 8,
        kernel_size: Sequence[int] = (2, 2, 2),
        stride: Sequence[int] = (2, 2, 2),
        seed: int = 0,
        temporal_padding: str = "none",
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        position_scale: float = 16.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.qk_dim = int(qk_dim)
        self.num_heads = int(num_heads)
        self.kernel_size = _as_3tuple(kernel_size, "representation.codec.kernel_size")
        self.stride = _as_3tuple(stride, "representation.codec.stride")
        self.seed = int(seed)
        self.temporal_padding = _validate_temporal_padding(temporal_padding)
        self.norm_eps = float(norm_eps)
        self.rope_base = float(rope_base)
        self.position_scale = float(position_scale)

        if self.input_dim <= 0 or self.output_dim <= 0 or self.qk_dim <= 0:
            raise ValueError(
                "FAE attention codec dimensions must be positive, "
                f"got input_dim={input_dim}, output_dim={output_dim}, qk_dim={qk_dim}."
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.kernel_size != (2, 2, 2) or self.stride != (2, 2, 2):
            raise ValueError(
                "FAEAttentionCausalCodec currently requires 2x2x2 kernel/stride, "
                f"got kernel={self.kernel_size}, stride={self.stride}."
            )
        if self.qk_dim % self.num_heads != 0:
            raise ValueError(
                f"qk_dim={self.qk_dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.output_dim % self.num_heads != 0:
            raise ValueError(
                f"output_dim={self.output_dim} must be divisible by num_heads={self.num_heads}."
            )
        self.qk_head_dim = self.qk_dim // self.num_heads
        self.value_head_dim = self.output_dim // self.num_heads
        if self.qk_head_dim % 4 != 0:
            raise ValueError(
                "2D RoPE requires qk_dim / num_heads to be divisible by 4, "
                f"got head_dim={self.qk_head_dim}."
            )
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {norm_eps}.")
        if self.rope_base <= 0 or self.position_scale <= 0:
            raise ValueError(
                "rope_base and position_scale must be positive, "
                f"got {rope_base} and {position_scale}."
            )

        self.patch_dim = self.input_dim * math.prod(self.kernel_size)
        self.q_projection = nn.Linear(self.patch_dim, self.qk_dim, bias=False)
        self.k_projection = nn.Linear(self.patch_dim, self.qk_dim, bias=False)
        self.value_projection = nn.Linear(self.patch_dim, self.output_dim, bias=False)
        self.output_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.input_norm = nn.RMSNorm(
            self.patch_dim,
            eps=self.norm_eps,
            elementwise_affine=False,
        )
        self.camera_q_bias = nn.Parameter(
            torch.empty(self.num_cameras, self.num_heads, self.qk_head_dim)
        )
        self.camera_k_bias = nn.Parameter(
            torch.empty(self.num_cameras, self.num_heads, self.qk_head_dim)
        )

        base_weight = _random_codec_weight(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            kernel_size=self.kernel_size,
            seed=self.seed,
        )
        with torch.no_grad():
            self.value_projection.weight.copy_(base_weight.flatten(1))
            self.output_projection.weight.zero_()
            nn.init.normal_(self.camera_q_bias, std=0.02)
            nn.init.normal_(self.camera_k_bias, std=0.02)

        axis_dim = self.qk_head_dim // 2
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / float(axis_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self._last_attention_entropy: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"qk_dim={self.qk_dim}, num_heads={self.num_heads}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, seed={self.seed}, "
            f"temporal_padding={self.temporal_padding!r}"
        )

    @property
    def last_attention_entropy(self) -> torch.Tensor | None:
        return self._last_attention_entropy

    def _validate_camera_features(self, camera_features: Sequence[torch.Tensor]) -> None:
        if len(camera_features) != self.num_cameras:
            raise ValueError(
                "FAEAttentionCausalCodec requires RobotWin front/left/right features, "
                f"got {len(camera_features)} tensors."
            )
        front, left, right = camera_features
        reference = front.shape[:3]
        for camera_idx, features in enumerate(camera_features):
            if features.ndim != 5:
                raise ValueError(
                    "FAEAttentionCausalCodec expects each camera as [B,C,T,H,W], "
                    f"camera{camera_idx}={tuple(features.shape)}."
                )
            if features.shape[:3] != reference:
                raise ValueError(
                    "FAE attention camera features must share [B,C,T], "
                    f"front={tuple(front.shape)} camera{camera_idx}={tuple(features.shape)}."
                )
            if int(features.shape[1]) != self.input_dim:
                raise ValueError(
                    f"Camera {camera_idx} has C={features.shape[1]}, expected {self.input_dim}."
                )
            expected_remainder = 1 if self.temporal_padding == "left_zero" else 0
            if int(features.shape[2]) % 2 != expected_remainder:
                raise ValueError(
                    f"temporal_padding={self.temporal_padding!r} requires T % 2 == "
                    f"{expected_remainder}, got camera{camera_idx} T={features.shape[2]}."
                )
            if int(features.shape[3]) % 2 or int(features.shape[4]) % 2:
                raise ValueError(
                    "FAE attention codec requires even camera feature grids, "
                    f"camera{camera_idx}={features.shape[-2:]}."
                )
        if left.shape[-2:] != right.shape[-2:]:
            raise ValueError(
                f"RobotWin wrist feature grids must match, got {left.shape[-2:]} and {right.shape[-2:]}."
            )
        if int(front.shape[-2]) != 2 * int(left.shape[-2]):
            raise ValueError(
                "RobotWin front feature height must be twice wrist height, "
                f"got front={front.shape[-2:]} wrist={left.shape[-2:]}."
            )
        if int(front.shape[-1]) != int(left.shape[-1]) + int(right.shape[-1]):
            raise ValueError(
                "RobotWin front feature width must equal both wrist widths, "
                f"got front={front.shape[-2:]} wrists={left.shape[-2:]}/{right.shape[-2:]}."
            )

    def _patchify_camera(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.temporal_padding == "left_zero":
            features = F.pad(features, (0, 0, 0, 0, 1, 0))
        patches = (
            features.unfold(2, 2, 2)
            .unfold(3, 2, 2)
            .unfold(4, 2, 2)
        )
        batch, channels, groups, height, width, _, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
            batch,
            groups,
            height * width,
            channels * 8,
        )
        return patches.contiguous(), (int(height), int(width))

    @staticmethod
    def _normalized_grid(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(height * width, 2)

    @staticmethod
    def _rotate_pairs(values: torch.Tensor) -> torch.Tensor:
        paired = values.unflatten(-1, (-1, 2))
        first, second = paired.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _apply_2d_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        axis_dim = self.qk_head_dim // 2
        inv_freq = self.rope_inv_freq.to(device=q.device, dtype=torch.float32)

        def apply_axis(values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
            angles = coordinate.float().unsqueeze(-1) * self.position_scale * inv_freq
            cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=values.dtype)
            sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=values.dtype)
            cos = cos.view(1, 1, positions.shape[0], axis_dim)
            sin = sin.view(1, 1, positions.shape[0], axis_dim)
            return values * cos + self._rotate_pairs(values) * sin

        q_y, q_x = q.split(axis_dim, dim=-1)
        k_y, k_x = k.split(axis_dim, dim=-1)
        q = torch.cat(
            (apply_axis(q_y, positions[:, 0]), apply_axis(q_x, positions[:, 1])),
            dim=-1,
        )
        k = torch.cat(
            (apply_axis(k_y, positions[:, 0]), apply_axis(k_x, positions[:, 1])),
            dim=-1,
        )
        return q, k

    def forward(self, camera_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        self._validate_camera_features(camera_features)
        camera_patches = []
        spatial_shapes = []
        camera_ids = []
        positions = []
        for camera_idx, features in enumerate(camera_features):
            patches, spatial_shape = self._patchify_camera(features)
            camera_patches.append(patches)
            spatial_shapes.append(spatial_shape)
            token_count = spatial_shape[0] * spatial_shape[1]
            camera_ids.append(
                torch.full((token_count,), camera_idx, device=features.device, dtype=torch.long)
            )
            positions.append(
                self._normalized_grid(
                    *spatial_shape,
                    device=features.device,
                    dtype=torch.float32,
                )
            )

        patches = torch.cat(camera_patches, dim=2)
        camera_ids_tensor = torch.cat(camera_ids, dim=0)
        positions_tensor = torch.cat(positions, dim=0)
        batch, groups, tokens, _ = patches.shape

        # Keep the normalized activation in the model dtype. Materializing an
        # explicit FP32 [B,G,N,8192] copy here would add hundreds of MB at
        # production batch sizes.
        normalized = self.input_norm(patches)

        q = self.q_projection(normalized).view(
            batch * groups, tokens, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)
        k = self.k_projection(normalized).view(
            batch * groups, tokens, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)
        base = self.value_projection(patches)
        value = base.view(
            batch * groups, tokens, self.num_heads, self.value_head_dim
        ).transpose(1, 2)

        q, k = self._apply_2d_rope(q, k, positions_tensor)
        camera_q_bias = self.camera_q_bias[camera_ids_tensor].permute(1, 0, 2)
        camera_k_bias = self.camera_k_bias[camera_ids_tensor].permute(1, 0, 2)
        q = q + camera_q_bias.unsqueeze(0).to(dtype=q.dtype)
        k = k + camera_k_bias.unsqueeze(0).to(dtype=k.dtype)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        mixed = torch.matmul(attention, value)
        mixed = mixed.transpose(1, 2).reshape(batch, groups, tokens, self.output_dim)
        latents = base + self.output_projection(mixed)

        entropy = -(attention.float().clamp_min(1e-12) * attention.float().clamp_min(1e-12).log())
        self._last_attention_entropy = (
            entropy.sum(dim=-1).mean() / math.log(max(tokens, 2))
        ).detach()

        outputs = []
        offset = 0
        for spatial_shape in spatial_shapes:
            height, width = spatial_shape
            token_count = height * width
            camera_latents = latents[:, :, offset : offset + token_count]
            camera_latents = camera_latents.view(
                batch, groups, height, width, self.output_dim
            ).permute(0, 4, 1, 2, 3)
            outputs.append(camera_latents.contiguous())
            offset += token_count
        return outputs


class CausalRunningChannelNorm(nn.Module):
    """Normalize codec latents with lagged per-channel running statistics.

    The module deliberately has no affine parameters. Unlike train-mode
    BatchNorm3d, the current forward uses the running statistics *before* they
    are updated from the current batch. Consequently, the normalized clean
    observation cannot depend on future frames from the same training clip.
    Statistics are updated after normalization and synchronized across an
    initialized torch.distributed process group.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.momentum = float(momentum)
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if not 0.0 < self.momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}.")

        self.register_buffer("running_mean", torch.zeros(self.num_channels, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(self.num_channels, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def _apply(self, fn):
        super()._apply(fn)
        # Accumulating running moments in BF16 is unnecessarily lossy. Preserve
        # their device placement while keeping floating-point buffers in FP32.
        self.running_mean = self.running_mean.float()
        self.running_var = self.running_var.float()
        return self

    def _validate_input(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5:
            raise ValueError(
                "CausalRunningChannelNorm expects [B,C,T,H,W], "
                f"got {tuple(latents.shape)}."
            )
        if int(latents.shape[1]) != self.num_channels:
            raise ValueError(
                "CausalRunningChannelNorm channel mismatch: "
                f"got {latents.shape[1]}, expected {self.num_channels}."
            )

    @staticmethod
    def _distributed_channel_moments(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = latents.detach().float()
        reduce_dims = (0, 2, 3, 4)
        channel_sum = values.sum(dim=reduce_dims)
        channel_sq_sum = values.square().sum(dim=reduce_dims)
        count = torch.tensor(
            values.numel() // values.shape[1],
            device=values.device,
            dtype=torch.float32,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(channel_sum)
            torch.distributed.all_reduce(channel_sq_sum)
            torch.distributed.all_reduce(count)

        batch_mean = channel_sum / count.clamp_min(1.0)
        batch_var = channel_sq_sum / count.clamp_min(1.0) - batch_mean.square()
        return batch_mean, batch_var.clamp_min(0.0)

    @torch.no_grad()
    def _update_running_stats(self, latents: torch.Tensor) -> None:
        batch_mean, batch_var = self._distributed_channel_moments(latents)
        momentum = self.momentum
        self.running_mean.lerp_(batch_mean.to(self.running_mean), momentum)
        self.running_var.lerp_(batch_var.to(self.running_var), momentum)
        self.num_batches_tracked.add_(1)

    def forward(
        self,
        latents: torch.Tensor,
        *,
        update_stats: bool | None = None,
    ) -> torch.Tensor:
        self._validate_input(latents)
        if update_stats is None:
            update_stats = self.training

        # Clone before the in-place EMA update so autograd never observes a
        # version change in buffers used by this forward.
        running_mean = self.running_mean.detach().clone().to(
            device=latents.device,
            dtype=torch.float32,
        )
        running_var = self.running_var.detach().clone().to(
            device=latents.device,
            dtype=torch.float32,
        )
        view_shape = (1, self.num_channels, 1, 1, 1)
        normalized = (
            latents.float() - running_mean.view(view_shape)
        ) / torch.sqrt(running_var.view(view_shape) + self.eps)

        if bool(update_stats):
            self._update_running_stats(latents)
        return normalized.to(dtype=latents.dtype).contiguous()

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_input(latents)
        running_mean = self.running_mean.to(device=latents.device, dtype=torch.float32)
        running_var = self.running_var.to(device=latents.device, dtype=torch.float32)
        view_shape = (1, self.num_channels, 1, 1, 1)
        raw = latents.float() * torch.sqrt(running_var.view(view_shape) + self.eps)
        raw = raw + running_mean.view(view_shape)
        return raw.to(dtype=latents.dtype).contiguous()


class CausalCodecFeatureDecoder(nn.Module):
    """Decode frozen causal codec latents back to their DINO feature grid.

    The frozen codec uses a 2x2x2 strided projection with one temporal left-pad.
    A matching transposed projection restores the pre-codec spatial resolution;
    dropping its first temporal output maps ``T`` codec steps to ``2*T-1`` DINO
    feature frames. This module intentionally stops at the DINO feature space;
    the released RAE decoder remains responsible for converting features to RGB.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        drop_first_temporal_output: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.drop_first_temporal_output = bool(drop_first_temporal_output)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError(
                "Decoder dimensions must be positive, "
                f"got input_dim={input_dim}, output_dim={output_dim}."
            )
        self.projection = nn.ConvTranspose3d(
            self.input_dim,
            self.output_dim,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(
                f"CausalCodecFeatureDecoder expects [B,C,T,H,W], got {tuple(latents.shape)}."
            )
        if int(latents.shape[1]) != self.input_dim:
            raise ValueError(
                "CausalCodecFeatureDecoder channel mismatch: "
                f"got {latents.shape[1]}, expected {self.input_dim}."
            )
        features = self.projection(latents)
        if self.drop_first_temporal_output:
            # The codec's temporal left pad creates an unused first member of
            # the first decoded pair. Removing it maps T steps to 2*T-1 frames.
            features = features[:, :, 1:]
        return features.contiguous()
