from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.models.wan22.wan_video_dit import create_group_causal_attn_mask


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


@dataclass(frozen=True)
class MotActionToWorldConfig:
    enabled: bool = False
    mask_mode: str = "group_diagonal"


def parse_mot_action_to_world_config(mot_conditioning: Optional[dict[str, Any]]) -> MotActionToWorldConfig:
    mot_conditioning = _as_plain_dict(mot_conditioning)
    action_to_world = _as_plain_dict(mot_conditioning.get("action_to_world", {}))
    mask_mode = str(action_to_world.get("mask_mode", "group_diagonal"))
    if mask_mode not in {"causal", "group_diagonal"}:
        raise ValueError(
            "`mot_conditioning.action_to_world.mask_mode` must be one of "
            "{'causal', 'group_diagonal'}, "
            f"got {mask_mode!r}."
        )
    return MotActionToWorldConfig(
        enabled=bool(action_to_world.get("enabled", False)),
        mask_mode=mask_mode,
    )


@torch.no_grad()
def build_world_action_mot_mask(
    *,
    world_expert,
    world_seq_len: int,
    action_seq_len: int,
    world_tokens_per_frame: int,
    device: torch.device,
    action_to_world: MotActionToWorldConfig,
) -> torch.Tensor:
    total_seq_len = int(world_seq_len) + int(action_seq_len)
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
    mask[:world_seq_len, :world_seq_len] = world_expert.build_video_to_video_mask(
        video_seq_len=world_seq_len,
        video_tokens_per_frame=world_tokens_per_frame,
        device=device,
    )
    mask[world_seq_len:, world_seq_len:] = True
    first_frame_tokens = min(world_tokens_per_frame, world_seq_len)
    mask[world_seq_len:, :first_frame_tokens] = True

    if action_to_world.enabled and world_seq_len > first_frame_tokens:
        if world_seq_len % world_tokens_per_frame != 0:
            raise ValueError(
                "`world_seq_len` must be divisible by `world_tokens_per_frame` for action-to-world attention, "
                f"got {world_seq_len} and {world_tokens_per_frame}."
            )
        num_world_frames = world_seq_len // world_tokens_per_frame
        num_future_frames = num_world_frames - 1
        if action_seq_len % num_future_frames != 0:
            raise ValueError(
                "Action sequence length must be divisible by future world frames for action-to-world attention, "
                f"got action_seq_len={action_seq_len}, future_frames={num_future_frames}."
            )
        action_group_mask = create_group_causal_attn_mask(
            num_temporal_groups=num_future_frames,
            num_query_per_group=world_tokens_per_frame,
            num_key_per_group=action_seq_len // num_future_frames,
            mode=action_to_world.mask_mode,
        ).to(device=device)
        mask[first_frame_tokens:world_seq_len, world_seq_len:] = action_group_mask
    return mask


def compute_action_flow_loss(
    *,
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: Optional[torch.Tensor],
    timestep_action: torch.Tensor,
    action_scheduler,
) -> torch.Tensor:
    action_loss_token = F.mse_loss(
        pred_action.float(),
        target_action.float(),
        reduction="none",
    ).mean(dim=2)
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
    else:
        action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = action_scheduler.training_weight(timestep_action).to(
        action_loss_per_sample.device,
        dtype=action_loss_per_sample.dtype,
    )
    return (action_loss_per_sample * action_weight).mean()
