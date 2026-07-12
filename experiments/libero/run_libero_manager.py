import os
import inspect
import hashlib
import json
import shlex
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf

from experiments.libero.task_sampling import (
    TaskCandidate,
    build_sampling_manifest,
    load_classification,
    stratified_sample,
)
from experiments.libero.persistent_scheduler import PersistentScheduler, visible_gpu_ids
from experiments.libero.summarize_results import summarize_results


def _default_classification_path() -> Path:
    benchmark_module_path = Path(inspect.getfile(benchmark)).resolve()
    return benchmark_module_path.with_name("task_classification.json")


def create_task_file(
    output_file: Path,
    task_suite_names: list[str],
    *,
    sample_ratio: float = 1.0,
    sample_seed: int = 42,
    classification_path: Path | None = None,
) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_candidates: list[TaskCandidate] = []
    for suite_name in task_suite_names:
        task_suite = benchmark_dict[suite_name]()
        n_tasks = int(task_suite.n_tasks)
        print(f"\n{suite_name}:")
        print(f"- Number of tasks: {n_tasks}")
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            all_candidates.append(
                TaskCandidate(suite=suite_name, task_id=task_id, name=str(task.name))
            )

    ratio = float(sample_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"MULTIRUN.sample_ratio must be in (0, 1], got {ratio}")
    selected_candidates = all_candidates
    if ratio < 1.0:
        resolved_classification_path = (
            _default_classification_path()
            if classification_path is None
            else classification_path.expanduser().resolve()
        )
        if not resolved_classification_path.exists():
            raise FileNotFoundError(
                "Stratified LIBERO-Plus sampling requires task_classification.json, "
                f"but it was not found at: {resolved_classification_path}"
            )
        classified, classification_sha256 = load_classification(
            resolved_classification_path, all_candidates
        )
        selected_candidates = stratified_sample(classified, ratio=ratio, seed=int(sample_seed))
        manifest = build_sampling_manifest(
            all_candidates=classified,
            sampled_candidates=selected_candidates,
            ratio=ratio,
            seed=int(sample_seed),
            classification_path=resolved_classification_path,
            classification_sha256=classification_sha256,
        )
        manifest_path = output_file.with_name(f"{output_file.stem}_sampling.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Sampled {len(selected_candidates)}/{len(all_candidates)} tasks "
            f"(ratio={ratio}, seed={sample_seed})."
        )
        print(f"Sampling manifest: {manifest_path}")

    with output_file.open("w", encoding="utf-8") as f:
        for candidate in selected_candidates:
            f.write(f"{candidate.suite},{candidate.task_id}\n")

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {len(selected_candidates)}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
        "EVALUATION.num_trials",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
    resume: bool,
    max_task_retries: int,
    task_timeout_seconds: int,
    worker_start_timeout_seconds: int,
    max_worker_restarts: int,
    worker_launch_interval_seconds: float,
    resolved_config_sha256: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    scheduler = PersistentScheduler(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=ckpt,
        gpu_ids=visible_gpu_ids(num_gpus),
        num_trials=num_trials,
        workers_per_gpu=max_tasks_per_gpu,
        output_dir=output_dir,
        extra_overrides=extra_overrides,
        resume=resume,
        max_task_retries=max_task_retries,
        task_timeout_seconds=task_timeout_seconds,
        worker_start_timeout_seconds=worker_start_timeout_seconds,
        max_worker_restarts=max_worker_restarts,
        worker_launch_interval_seconds=worker_launch_interval_seconds,
        resolved_config_sha256=resolved_config_sha256,
    )
    try:
        scheduler.run()
    finally:
        if any(output_dir.glob("*/gpu*_task*_results.json")):
            print("Generating evaluation report from available results...")
            summarize_results(str(output_dir))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:
        task_file = output_dir / "tasks.txt"
    classification_path_cfg = manager.get("task_classification_path")
    classification_path = None
    if classification_path_cfg is not None:
        classification_path = Path(
            os.path.expanduser(os.path.expandvars(str(classification_path_cfg)))
        )
    task_file = create_task_file(
        task_file,
        list(manager.task_suite_names),
        sample_ratio=float(manager.get("sample_ratio", 1.0)),
        sample_seed=int(manager.get("sample_seed", 42)),
        classification_path=classification_path,
    )

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
        resume=bool(manager.get("resume", True)),
        max_task_retries=int(manager.get("max_task_retries", 1)),
        task_timeout_seconds=int(manager.get("task_timeout_seconds", 1800)),
        worker_start_timeout_seconds=int(manager.get("worker_start_timeout_seconds", 900)),
        max_worker_restarts=int(manager.get("max_worker_restarts", 3)),
        worker_launch_interval_seconds=float(manager.get("worker_launch_interval_seconds", 1.0)),
        resolved_config_sha256=hashlib.sha256(
            OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
        ).hexdigest(),
    )


if __name__ == "__main__":
    main()
