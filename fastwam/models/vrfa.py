from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.models.representation_forcing import RepresentationForcing
from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class VrfA(FastWAM):
    """FastWAM with representation forcing on the video expert.

    This class keeps FastWAM inference/deploy APIs intact and only changes the
    training objective by adding REPA, Geometry Forcing, or ReDi/DreamWorld-style
    representation supervision.
    """

    def __init__(
        self,
        *args,
        representation_forcing: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.capability = {
            "video_inference": True,
            "vae_reconstruction": True,
            "action_inference": True,
        }
        self.capabilities = self.capability
        self._capture_representation_viz = False
        self._last_representation_viz = None
        self._init_representation_forcing(representation_forcing)

    def _init_representation_forcing(self, representation_forcing: Optional[dict[str, Any]]):
        self.representation_forcing = RepresentationForcing(
            hidden_dim=int(self.video_expert.hidden_dim),
            cfg=representation_forcing,
        ).to(device=self.device, dtype=self.torch_dtype)
        self._validate_representation_forcing_config()
        return self.representation_forcing

    def _validate_representation_forcing_config(self):
        repr_forcing = self.representation_forcing
        if repr_forcing is None or not repr_forcing.enabled:
            return
        invalid_layers = [
            layer
            for layer in repr_forcing.layer_indices
            if layer < 0 or layer >= int(self.mot.num_layers)
        ]
        if invalid_layers:
            raise ValueError(
                "`representation_forcing.layer_indices` contains invalid MoT layer indices: "
                f"{invalid_layers}. Valid range is [0, {int(self.mot.num_layers) - 1}]."
            )

    def set_representation_visualization_capture(self, enabled: bool):
        self._capture_representation_viz = bool(enabled)
        if not enabled:
            self._last_representation_viz = None

    def pop_last_representation_visualization(self):
        payload = self._last_representation_viz
        self._last_representation_viz = None
        return payload

    @staticmethod
    def _tokens_to_pca_rgb_strip(
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        f, h, w = [int(v) for v in grid_size]
        expected_tokens = f * h * w
        if pred_tokens.ndim != 3 or target_tokens.ndim != 3:
            raise ValueError(
                "Representation visualization expects pred/target tokens [B,N,C], "
                f"got {tuple(pred_tokens.shape)} and {tuple(target_tokens.shape)}."
            )
        if pred_tokens.shape[1] != expected_tokens or target_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "Representation visualization token count mismatch: "
                f"pred={pred_tokens.shape[1]} target={target_tokens.shape[1]} expected={expected_tokens}."
            )

        pred = pred_tokens[0].detach().float().cpu()
        target = target_tokens[0].detach().float().cpu()
        if pred.shape[-1] != target.shape[-1]:
            raise ValueError(
                "Representation visualization channel mismatch: "
                f"pred={pred.shape[-1]} target={target.shape[-1]}."
            )

        combined = torch.cat([target, pred], dim=0)
        combined = combined - combined.mean(dim=0, keepdim=True)
        q = min(3, int(combined.shape[0] - 1), int(combined.shape[1]))
        if q <= 0:
            raise ValueError(f"Cannot PCA-project representation tokens with shape {tuple(combined.shape)}.")
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
        target_rgb, pred_rgb = projected[:expected_tokens], projected[expected_tokens:]
        target_rgb = target_rgb.reshape(f, h, w, 3).permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.reshape(f, h, w, 3).permute(0, 3, 1, 2)
        diff_rgb = (pred_rgb - target_rgb).abs()

        rows = []
        for frame_idx in range(f):
            rows.append(torch.cat([target_rgb[frame_idx], pred_rgb[frame_idx], diff_rgb[frame_idx]], dim=1))
        return torch.cat(rows, dim=2).contiguous()

    def _maybe_store_representation_visualization(
        self,
        *,
        repr_forcing: RepresentationForcing,
        target_tokens: torch.Tensor,
        final_video_tokens: torch.Tensor,
        intermediate_video_tokens: Optional[dict[int, torch.Tensor]],
        redi_state: Optional[dict[str, torch.Tensor]],
        grid_size: tuple[int, int, int],
    ):
        if not self._capture_representation_viz:
            return
        self._capture_representation_viz = False
        self._last_representation_viz = None
        with torch.no_grad():
            tokens = repr_forcing.visualization_tokens(
                target_tokens=target_tokens,
                final_video_tokens=final_video_tokens,
                intermediate_video_tokens=intermediate_video_tokens,
                redi_state=redi_state,
            )
            if tokens is None:
                return
            name, pred_tokens, target_tokens = tokens
            image = self._tokens_to_pca_rgb_strip(
                pred_tokens=pred_tokens,
                target_tokens=target_tokens,
                grid_size=grid_size,
            )
        self._last_representation_viz = {
            "tag": f"train/representation_pca/{name}",
            "image": image,
        }

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        representation_forcing: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        return super().from_wan22_pretrained(
            *args,
            representation_forcing=representation_forcing,
            **kwargs,
        )

    def training_loss(self, sample, tiled: bool = False):
        repr_forcing = self.representation_forcing
        inputs = self.build_inputs(
            sample,
            tiled=tiled,
            return_input_video=bool(repr_forcing.enabled),
        )
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        repr_target_tokens = None
        redi_state = None
        if repr_forcing.enabled:
            input_video = inputs.pop("input_video", None)
            if input_video is None:
                raise ValueError("VrfA expected `input_video` from build_inputs when representation forcing is enabled.")
            repr_target_tokens = repr_forcing.get_target_tokens(
                input_video=input_video,
                grid_size=video_pre["meta"]["grid_size"],
                device=video_tokens.device,
                dtype=video_tokens.dtype,
            )
            del input_video
            video_tokens, redi_state = repr_forcing.apply_redi_input(
                video_tokens=video_tokens,
                target_tokens=repr_target_tokens,
                timestep=timestep_video,
                scheduler=self.train_video_scheduler,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        repr_layers = (
            set(repr_forcing.layer_indices)
            if repr_forcing.requires_intermediate_layers
            else None
        )
        mot_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            return_intermediate_layers=repr_layers,
        )
        if repr_layers is not None:
            tokens_out, intermediate_tokens = mot_out
            missing_repr_layers = sorted(repr_layers - set(intermediate_tokens.keys()))
            if missing_repr_layers:
                raise ValueError(
                    "MoT did not return requested representation forcing layers: "
                    f"{missing_repr_layers}. Returned layers: {sorted(intermediate_tokens.keys())}."
                )
            intermediate_video_tokens = {
                layer: payload["video"] for layer, payload in intermediate_tokens.items()
            }
        else:
            tokens_out = mot_out
            intermediate_video_tokens = None

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

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

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_repr = pred_video.new_zeros(())
        repr_loss_dict = {}
        if repr_forcing.enabled:
            if repr_target_tokens is None:
                raise ValueError("Representation forcing target tokens were not prepared.")
            loss_repr, repr_loss_dict = repr_forcing.compute_loss(
                target_tokens=repr_target_tokens,
                final_video_tokens=tokens_out["video"],
                intermediate_video_tokens=intermediate_video_tokens,
                redi_state=redi_state,
            )
            self._maybe_store_representation_visualization(
                repr_forcing=repr_forcing,
                target_tokens=repr_target_tokens,
                final_video_tokens=tokens_out["video"],
                intermediate_video_tokens=intermediate_video_tokens,
                redi_state=redi_state,
                grid_size=video_pre["meta"]["grid_size"],
            )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action + loss_repr
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if repr_loss_dict:
            for key, value in repr_loss_dict.items():
                loss_dict[key] = repr_forcing.loss_weight * float(value)
        return loss_total, loss_dict

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "representation_forcing": self.representation_forcing.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "model_class": "VrfA",
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        if "representation_forcing" in payload:
            self.representation_forcing.load_state_dict(payload["representation_forcing"], strict=False)
        else:
            logger.warning("Checkpoint has no `representation_forcing` weights; keeping current VrfA heads.")

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

    @torch.no_grad()
    def _predict_joint_noise_with_redi(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        latents_repr: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_tokens = self.representation_forcing.apply_redi_inference_input(
            video_tokens=video_pre["tokens"],
            representation_latents=latents_repr,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_repr = self.representation_forcing.predict_redi_velocity(tokens_out["video"])
        return pred_video, pred_action, pred_repr

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, action_cfg_scale
        if not (
            getattr(self, "representation_forcing", None) is not None
            and self.representation_forcing.enabled
            and self.representation_forcing.mode == "redi"
        ):
            return super().infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                test_action_with_infer_action=test_action_with_infer_action,
            )

        self.eval()
        num_video_frames = int(num_video_frames)
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            expected_action_dim = int(getattr(self.video_expert, "action_dim", self.action_expert.action_dim))
            if (
                action.ndim != 3
                or action.shape[0] != 1
                or action.shape[1] != action_horizon
                or action.shape[2] != expected_action_dim
            ):
                raise ValueError(
                    "`action` must have shape [1, action_horizon, action_dim] or [action_horizon, action_dim], "
                    f"got {tuple(action.shape)} with action_horizon={action_horizon}, "
                    f"action_dim={expected_action_dim}."
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        repr_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed + 17)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        patch_t, patch_h, patch_w = [int(v) for v in self.video_expert.patch_size]
        if latent_t % patch_t != 0 or latent_h % patch_h != 0 or latent_w % patch_w != 0:
            raise ValueError(
                "Latent grid must be divisible by video DiT patch size for ReDi inference: "
                f"latent_grid={(latent_t, latent_h, latent_w)} patch_size={(patch_t, patch_h, patch_w)}"
            )
        repr_seq_len = (latent_t // patch_t) * (latent_h // patch_h) * (latent_w // patch_w)
        latents_repr = torch.randn(
            (1, repr_seq_len, self.representation_forcing.target_dim),
            generator=repr_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            pred_video, pred_action, pred_repr = self._predict_joint_noise_with_redi(
                latents_video=latents_video,
                latents_action=latents_action,
                latents_repr=latents_repr,
                timestep_video=step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device),
                timestep_action=step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device),
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_repr = self.infer_video_scheduler.step(pred_repr, step_delta_video, latents_repr)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action and not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
            logger.warning("Action from VrfA ReDi infer_joint differs from infer_action.")
        return {"video": self._decode_latents(latents_video, tiled=tiled), "action": action_out}
