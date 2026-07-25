from __future__ import annotations

from typing import Any, Optional

import torch

from .ra import RA


class RAJoint(RA):
    """RA variant that jointly denoises future representations and actions.

    As in :class:`FastWAMJoint`, the world/representation stream remains
    independent of the action stream, while every action query may attend to
    all current and noisy future representation tokens. Consequently,
    inference must run both diffusion streams together.
    """

    @classmethod
    def from_config(cls, **kwargs):
        representation_dit_config = kwargs.get("representation_dit_config")
        if not isinstance(representation_dit_config, dict):
            raise ValueError(
                "`representation_dit_config` must be provided as dict for RAJoint."
            )
        if bool(representation_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "RAJoint requires `representation_dit_config['action_conditioned']=false`."
            )

        mot_conditioning = kwargs.get("mot_conditioning") or {}
        action_to_world = mot_conditioning.get("action_to_world", {})
        if bool(action_to_world.get("enabled", False)):
            raise ValueError(
                "RAJoint keeps representation prediction action-independent; "
                "disable `mot_conditioning.action_to_world`."
            )
        return super().from_config(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = int(video_seq_len) + int(action_seq_len)
        mask = torch.zeros(
            (total_seq_len, total_seq_len),
            dtype=torch.bool,
            device=device,
        )
        mask[:video_seq_len, :video_seq_len] = self.representation_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

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
                    "RAJoint `representation.temporal_groups` exceeds inference video length: "
                    f"groups={self.temporal_groups}, num_video_frames={num_video_frames}."
                )
            return len(self.temporal_groups)
        if self.temporal_indices is not None:
            max_index = max(self.temporal_indices)
            if max_index >= num_video_frames:
                raise ValueError(
                    "RAJoint `representation.temporal_indices` exceeds inference video length: "
                    f"indices={self.temporal_indices}, num_video_frames={num_video_frames}."
                )
            return len(self.temporal_indices)
        return num_video_frames

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
        return pred_repr, pred_action

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
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
        del negative_prompt, text_cfg_scale, tiled
        self.eval()
        if num_video_frames is None:
            raise ValueError(
                "RAJoint.infer_action requires `num_video_frames` to determine the future representation horizon."
            )
        num_repr_steps = self._num_inference_representation_steps(num_video_frames)

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            batch_size=1,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_repr = self._encode_inference_first_frame_latents(input_image)
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
        for step_t_repr, step_delta_repr, step_t_action, step_delta_action in zip(
            infer_timesteps_repr,
            infer_deltas_repr,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_repr = step_t_repr.unsqueeze(0).to(
                device=self.device,
                dtype=latents_repr.dtype,
            )
            timestep_action = step_t_action.unsqueeze(0).to(
                device=self.device,
                dtype=latents_action.dtype,
            )
            pred_repr, pred_action = self._predict_joint_noise(
                latents_repr=latents_repr,
                latents_action=latents_action,
                timestep_repr=timestep_repr,
                timestep_action=timestep_action,
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

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
