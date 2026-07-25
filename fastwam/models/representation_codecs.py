from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_3tuple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in value)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"`{name}` must contain three positive integers, got {values}.")
    return values


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
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_size = _as_3tuple(kernel_size, "representation.codec.kernel_size")
        self.stride = _as_3tuple(stride, "representation.codec.stride")
        self.seed = int(seed)
        self.norm_eps = float(norm_eps)

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

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        weight = torch.randn(
            self.output_dim,
            self.input_dim,
            *self.kernel_size,
            generator=generator,
            dtype=torch.float32,
        )
        # Unit-norm rows preserve unit input variance without changing the
        # process-global RNG state. The projection remains frozen as a buffer.
        weight = F.normalize(weight.flatten(1), dim=1).reshape_as(weight)
        self.register_buffer("weight", weight, persistent=True)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, seed={self.seed}"
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
        if int(features.shape[2]) % 2 != 1:
            raise ValueError(
                "FrozenRandomCausalCodec requires an odd number of frames so the first "
                f"latent is a singleton, got T={features.shape[2]}."
            )
        if int(features.shape[3]) % 2 != 0 or int(features.shape[4]) % 2 != 0:
            raise ValueError(
                "FrozenRandomCausalCodec requires even spatial dimensions, "
                f"got HxW={features.shape[3]}x{features.shape[4]}."
            )

        # Conv3d pads are ordered W-left/right, H-left/right, T-left/right.
        features = F.pad(features, (0, 0, 0, 0, 1, 0))
        latents = F.conv3d(
            features,
            self.weight.to(device=features.device, dtype=features.dtype),
            bias=None,
            stride=self.stride,
        )
        # Match the useful normalization from the previous compact bottleneck,
        # without introducing trainable affine parameters into the frozen codec.
        latents = latents.permute(0, 2, 3, 4, 1)
        latents = F.layer_norm(latents.float(), (self.output_dim,), eps=self.norm_eps)
        return latents.permute(0, 4, 1, 2, 3).to(dtype=features.dtype).contiguous()

    def expected_output_shape(self, input_shape: Sequence[int]) -> tuple[int, int, int, int, int]:
        if len(input_shape) != 5:
            raise ValueError(f"Expected a 5D input shape, got {tuple(input_shape)}.")
        batch, channels, frames, height, width = [int(v) for v in input_shape]
        if channels != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got C={channels}.")
        return batch, self.output_dim, math.ceil(frames / 2), height // 2, width // 2


class LearnedCausalCodec(nn.Module):
    """Learned causal projection with the same layout as the random codec."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        kernel_size: Sequence[int] = (2, 2, 2),
        stride: Sequence[int] = (2, 2, 2),
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_size = _as_3tuple(kernel_size, "representation.codec.kernel_size")
        self.stride = _as_3tuple(stride, "representation.codec.stride")
        self.norm_eps = float(norm_eps)

        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError(
                "representation codec dimensions must be positive, "
                f"got input_dim={self.input_dim}, output_dim={self.output_dim}."
            )
        if self.kernel_size != (2, 2, 2) or self.stride != (2, 2, 2):
            raise ValueError(
                "LearnedCausalCodec currently requires 2x2x2 kernel/stride, "
                f"got kernel={self.kernel_size}, stride={self.stride}."
            )
        self.projection = nn.Conv3d(
            self.input_dim,
            self.output_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(f"LearnedCausalCodec expects [B,C,T,H,W], got {tuple(features.shape)}.")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"LearnedCausalCodec channel mismatch: got {features.shape[1]}, expected {self.input_dim}."
            )
        if int(features.shape[2]) % 2 != 1:
            raise ValueError(
                "LearnedCausalCodec requires an odd number of frames, "
                f"got T={features.shape[2]}."
            )
        if int(features.shape[3]) % 2 != 0 or int(features.shape[4]) % 2 != 0:
            raise ValueError(
                "LearnedCausalCodec requires even spatial dimensions, "
                f"got HxW={features.shape[3]}x{features.shape[4]}."
            )

        features = F.pad(features, (0, 0, 0, 0, 1, 0))
        latents = self.projection(features)
        latents = latents.permute(0, 2, 3, 4, 1)
        latents = F.layer_norm(latents.float(), (self.output_dim,), eps=self.norm_eps)
        return latents.permute(0, 4, 1, 2, 3).to(dtype=features.dtype).contiguous()

    def expected_output_shape(self, input_shape: Sequence[int]) -> tuple[int, int, int, int, int]:
        if len(input_shape) != 5:
            raise ValueError(f"Expected a 5D input shape, got {tuple(input_shape)}.")
        batch, channels, frames, height, width = [int(v) for v in input_shape]
        if channels != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got C={channels}.")
        return batch, self.output_dim, math.ceil(frames / 2), height // 2, width // 2


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
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
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
        # The codec's temporal left pad creates an unused first member of the
        # first decoded pair. Removing it maps T codec steps to 2*T-1 frames.
        return features[:, :, 1:].contiguous()


def load_codec_weights(
    checkpoint: str | Path | dict[str, Any],
    *,
    encoder: nn.Module,
    decoder: Optional[nn.Module] = None,
    component: Optional[str] = None,
) -> dict[str, Any]:
    """Strictly load one component from a standalone codec checkpoint.

    Bundled ``input_mode=all`` checkpoints store keys such as
    ``front.projection.weight``. Passing ``component='front'`` strips that
    prefix before loading the standalone encoder/decoder modules.
    """
    if isinstance(checkpoint, dict):
        payload = checkpoint
        checkpoint_name = "<in-memory codec checkpoint>"
    else:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Codec checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        checkpoint_name = str(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Codec checkpoint must be a dict, got {type(payload)!r}: {checkpoint_name}"
        )

    checkpoint_input_dim = payload.get("input_dim")
    checkpoint_output_dim = payload.get("output_dim")
    if checkpoint_input_dim is not None and int(checkpoint_input_dim) != int(
        getattr(encoder, "input_dim", -1)
    ):
        raise ValueError(
            "Codec checkpoint input_dim mismatch: "
            f"checkpoint={checkpoint_input_dim}, model={getattr(encoder, 'input_dim', None)}."
        )
    if checkpoint_output_dim is not None and int(checkpoint_output_dim) != int(
        getattr(encoder, "output_dim", -1)
    ):
        raise ValueError(
            "Codec checkpoint output_dim mismatch: "
            f"checkpoint={checkpoint_output_dim}, model={getattr(encoder, 'output_dim', None)}."
        )

    encoder_state = payload.get("codec_encoder", payload.get("representation_codec"))
    decoder_state = payload.get("codec_decoder")
    if not isinstance(encoder_state, dict):
        raise ValueError(f"Codec checkpoint has no valid `codec_encoder`: {checkpoint_name}")
    if decoder is not None and not isinstance(decoder_state, dict):
        raise ValueError(f"Codec checkpoint has no valid `codec_decoder`: {checkpoint_name}")

    input_mode = payload.get("input_mode")
    if component is not None:
        component = str(component).lower()
        if input_mode == "all":
            prefix = f"{component}."

            def select_component(state: dict[str, torch.Tensor], state_name: str):
                selected = {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                }
                if not selected:
                    raise ValueError(
                        f"Bundled codec checkpoint has no {component!r} {state_name}: "
                        f"{checkpoint_name}"
                    )
                return selected

            encoder_state = select_component(encoder_state, "encoder")
            if decoder is not None:
                decoder_state = select_component(decoder_state, "decoder")
        elif input_mode not in (None, component):
            raise ValueError(
                f"Expected codec component {component!r}, got input_mode={input_mode!r}: "
                f"{checkpoint_name}"
            )

    encoder.load_state_dict(encoder_state, strict=True)
    if decoder is not None:
        decoder.load_state_dict(decoder_state, strict=True)
    return payload
