import logging
import math
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def _as_resolved_dict(value, name: str, *, default=None, required: bool = False):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required.")
        value = {} if default is None else default
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def _validate_action_scheduler(action_scheduler: dict, model_name: str):
    required_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys for {model_name}: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _resolve_rarae_shift(
    *,
    value,
    representation: dict,
    representation_dit_config: dict,
    shift_base_dim: int,
    shift_scope: str,
    name: str,
) -> float:
    if not _is_auto(value):
        return float(value)
    encoder_cfg = _as_resolved_dict(representation.get("encoder", {}), "representation.encoder")
    channels = int(
        representation.get(
            "target_dim",
            encoder_cfg.get("output_dim", representation_dit_config.get("in_dim", 1024)),
        )
    )
    latent_spatial_size = representation.get("latent_spatial_size", [12, 10])
    if isinstance(latent_spatial_size, int):
        height = width = int(latent_spatial_size)
    else:
        height, width = int(latent_spatial_size[0]), int(latent_spatial_size[1])
    temporal_groups = representation.get("temporal_groups", None)
    temporal_indices = representation.get("temporal_indices", None)
    if temporal_groups not in (None, "", "null"):
        steps = len(temporal_groups)
    elif temporal_indices not in (None, "", "null"):
        steps = len(temporal_indices)
    else:
        steps = 3
    scope = str(shift_scope).lower()
    if scope == "future_tokens":
        time_steps = max(int(steps) - 1, 1)
    elif scope == "all_tokens":
        time_steps = max(int(steps), 1)
    else:
        raise ValueError(f"Unsupported RARAE representation_scheduler.shift_scope: {shift_scope!r}.")
    if shift_base_dim <= 0:
        raise ValueError(f"RARAE representation_scheduler.shift_base_dim must be positive, got {shift_base_dim}.")
    latent_dim = channels * time_steps * height * width
    shift = math.sqrt(float(latent_dim) / float(shift_base_dim))
    logger.info(
        "Resolved RARAE %s=auto to %.4f from latent_dim=%d "
        "(C=%d T=%d H=%d W=%d base=%d scope=%s).",
        name,
        shift,
        latent_dim,
        channels,
        time_steps,
        height,
        width,
        shift_base_dim,
        scope,
    )
    return float(shift)


def _fastwam_pretrained_kwargs(
    *,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int,
    load_text_encoder: bool,
    proprio_dim: int | None,
    action_dit_config,
    action_dit_pretrained_path: str | None,
    skip_dit_load_from_pretrain: bool,
    video_scheduler,
    action_scheduler,
    loss,
    mot_checkpoint_mixed_attn: bool,
    redirect_common_files: bool,
    model_dtype: torch.dtype,
    device: str,
    model_name: str,
):
    video_dit_config = _as_resolved_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = _as_resolved_dict(action_dit_config, "action_dit_config")
    video_scheduler = _as_resolved_dict(video_scheduler, "video_scheduler")
    action_scheduler = _as_resolved_dict(action_scheduler, "action_scheduler", required=True)
    loss = _as_resolved_dict(loss, "loss")
    _validate_action_scheduler(action_scheduler, model_name)

    return {
        "device": device,
        "torch_dtype": model_dtype,
        "model_id": model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "tokenizer_max_len": int(tokenizer_max_len),
        "load_text_encoder": bool(load_text_encoder),
        "proprio_dim": None if proprio_dim is None else int(proprio_dim),
        "redirect_common_files": bool(redirect_common_files),
        "video_dit_config": video_dit_config,
        "action_dit_config": action_dit_config,
        "action_dit_pretrained_path": action_dit_pretrained_path,
        "skip_dit_load_from_pretrain": bool(skip_dit_load_from_pretrain),
        "mot_checkpoint_mixed_attn": bool(mot_checkpoint_mixed_attn),
        "video_train_shift": float(video_scheduler.get("train_shift", 5.0)),
        "video_infer_shift": float(video_scheduler.get("infer_shift", 5.0)),
        "video_num_train_timesteps": int(video_scheduler.get("num_train_timesteps", 1000)),
        "action_train_shift": float(action_scheduler["train_shift"]),
        "action_infer_shift": float(action_scheduler["infer_shift"]),
        "action_num_train_timesteps": int(action_scheduler["num_train_timesteps"]),
        "loss_lambda_video": float(loss.get("lambda_video", 1.0)),
        "loss_lambda_action": float(loss.get("lambda_action", 1.0)),
    }


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.fastwam import FastWAM

    return FastWAM.from_wan22_pretrained(
        **_fastwam_pretrained_kwargs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            video_dit_config=video_dit_config,
            tokenizer_max_len=tokenizer_max_len,
            load_text_encoder=load_text_encoder,
            proprio_dim=proprio_dim,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            loss=loss,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            redirect_common_files=redirect_common_files,
            model_dtype=model_dtype,
            device=device,
            model_name="FastWAM",
        )
    )


def create_vrfa(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    representation_forcing=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.vrfa import VrfA

    representation_forcing = _as_resolved_dict(representation_forcing, "representation_forcing")

    return VrfA.from_wan22_pretrained(
        **_fastwam_pretrained_kwargs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            video_dit_config=video_dit_config,
            tokenizer_max_len=tokenizer_max_len,
            load_text_encoder=load_text_encoder,
            proprio_dim=proprio_dim,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            loss=loss,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            redirect_common_files=redirect_common_files,
            model_dtype=model_dtype,
            device=device,
            model_name="VrfA",
        ),
        representation_forcing=representation_forcing,
    )


def create_ra(
    model_id: str,
    tokenizer_model_id: str,
    representation_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    representation_dit_pretrained_source: str | None = None,
    representation_dit_pretrained_path: str | None = None,
    representation_dit_pretrained_model_id: str | None = None,
    # Consumed by Wan22Trainer from cfg.model; accepted here because Hydra passes
    # all model config keys into this target.
    representation_dit_lr_scale: float | None = None,
    representation_scheduler=None,
    action_scheduler=None,
    loss=None,
    representation=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.helpers.loader import load_wan22_text_components
    from .models.ra import RA

    representation_dit_config = _as_resolved_dict(
        representation_dit_config,
        "representation_dit_config",
        required=True,
    )
    action_dit_config = _as_resolved_dict(action_dit_config, "action_dit_config")
    representation_scheduler = _as_resolved_dict(representation_scheduler, "representation_scheduler")
    action_scheduler = _as_resolved_dict(action_scheduler, "action_scheduler", required=True)
    loss = _as_resolved_dict(loss, "loss")
    representation = _as_resolved_dict(representation, "representation")
    _validate_action_scheduler(action_scheduler, "RA")

    text_components = load_wan22_text_components(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        load_text_encoder=bool(load_text_encoder),
    )

    model = RA.from_config(
        representation_dit_config=representation_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        representation_dit_pretrained_source=representation_dit_pretrained_source,
        representation_dit_pretrained_path=representation_dit_pretrained_path,
        representation_dit_pretrained_model_id=representation_dit_pretrained_model_id or model_id,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        representation=representation,
        text_encoder=text_components.text_encoder,
        tokenizer=text_components.tokenizer,
        text_dim=int(representation_dit_config["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        representation_train_shift=float(representation_scheduler.get("train_shift", 5.0)),
        representation_infer_shift=float(representation_scheduler.get("infer_shift", 5.0)),
        representation_num_train_timesteps=int(representation_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_representation=float(loss.get("lambda_representation", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    model.model_paths.update(
        {
            "text_encoder": text_components.text_encoder_path,
            "tokenizer": text_components.tokenizer_path,
        }
    )
    return model


def create_rarae(
    model_id: str,
    tokenizer_model_id: str,
    representation_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    representation_dit_pretrained_source: str | None = None,
    representation_dit_pretrained_path: str | None = None,
    representation_dit_pretrained_model_id: str | None = None,
    representation_dit_lr_scale: float | None = None,
    representation_scheduler=None,
    action_scheduler=None,
    loss=None,
    representation=None,
    representation_prediction=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    del representation_dit_lr_scale
    from .models.helpers.loader import load_wan22_text_components
    from .models.ra import RARAE

    representation_dit_config = _as_resolved_dict(
        representation_dit_config,
        "representation_dit_config",
        required=True,
    )
    action_dit_config = _as_resolved_dict(action_dit_config, "action_dit_config")
    representation_scheduler = _as_resolved_dict(representation_scheduler, "representation_scheduler")
    action_scheduler = _as_resolved_dict(action_scheduler, "action_scheduler", required=True)
    loss = _as_resolved_dict(loss, "loss")
    representation = _as_resolved_dict(representation, "representation")
    representation_prediction = _as_resolved_dict(
        representation_prediction,
        "representation_prediction",
        default={"type": "x", "x_loss_mode": "direct", "t_eps": 0.05},
    )
    _validate_action_scheduler(action_scheduler, "RARAE")

    shift_base_dim = int(representation_scheduler.get("shift_base_dim", 4096))
    shift_scope = str(representation_scheduler.get("shift_scope", "future_tokens"))
    representation_train_shift = _resolve_rarae_shift(
        value=representation_scheduler.get("train_shift", "auto"),
        representation=representation,
        representation_dit_config=representation_dit_config,
        shift_base_dim=shift_base_dim,
        shift_scope=shift_scope,
        name="train_shift",
    )
    representation_infer_shift = _resolve_rarae_shift(
        value=representation_scheduler.get("infer_shift", "auto"),
        representation=representation,
        representation_dit_config=representation_dit_config,
        shift_base_dim=shift_base_dim,
        shift_scope=shift_scope,
        name="infer_shift",
    )

    text_components = load_wan22_text_components(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        load_text_encoder=bool(load_text_encoder),
    )

    model = RARAE.from_config(
        representation_dit_config=representation_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        representation_dit_pretrained_source=representation_dit_pretrained_source,
        representation_dit_pretrained_path=representation_dit_pretrained_path,
        representation_dit_pretrained_model_id=representation_dit_pretrained_model_id or model_id,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        representation=representation,
        representation_prediction=representation_prediction,
        text_encoder=text_components.text_encoder,
        tokenizer=text_components.tokenizer,
        text_dim=int(representation_dit_config["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        representation_train_shift=representation_train_shift,
        representation_infer_shift=representation_infer_shift,
        representation_num_train_timesteps=int(representation_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_representation=float(loss.get("lambda_representation", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    model.model_paths.update(
        {
            "text_encoder": text_components.text_encoder_path,
            "tokenizer": text_components.tokenizer_path,
        }
    )
    return model


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.fastwam_joint import FastWAMJoint

    return FastWAMJoint.from_wan22_pretrained(
        **_fastwam_pretrained_kwargs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            video_dit_config=video_dit_config,
            tokenizer_max_len=tokenizer_max_len,
            load_text_encoder=load_text_encoder,
            proprio_dim=proprio_dim,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            loss=loss,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            redirect_common_files=redirect_common_files,
            model_dtype=model_dtype,
            device=device,
            model_name="FastWAMJoint",
        )
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.fastwam_idm import (
        FastWAMIDM,
    )

    return FastWAMIDM.from_wan22_pretrained(
        **_fastwam_pretrained_kwargs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            video_dit_config=video_dit_config,
            tokenizer_max_len=tokenizer_max_len,
            load_text_encoder=load_text_encoder,
            proprio_dim=proprio_dim,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            loss=loss,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            redirect_common_files=redirect_common_files,
            model_dtype=model_dtype,
            device=device,
            model_name="FastWAMIDM",
        )
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
