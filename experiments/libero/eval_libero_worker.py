"""Persistent LIBERO evaluation worker controlled through JSONL stdin events."""

import json
import os
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig

from experiments.libero.eval_libero_single import build_evaluation_runtime, evaluate_task


def _event_writer():
    raw_fd = os.environ.get("LIBERO_EVENT_FD")
    if raw_fd is None:
        raise RuntimeError("LIBERO_EVENT_FD is required for persistent worker mode.")
    return os.fdopen(int(raw_fd), "w", encoding="utf-8", buffering=1, closefd=True)


def _emit(stream, event: str, **payload) -> None:
    stream.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
    stream.flush()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    events = _event_writer()
    worker_id = str(os.environ["LIBERO_WORKER_ID"])
    gpu_id = int(os.environ["LIBERO_GPU_ID"])
    try:
        runtime = build_evaluation_runtime(cfg)
        _emit(
            events,
            "READY",
            worker_id=worker_id,
            gpu_id=gpu_id,
            model_load_seconds=runtime.model_load_seconds,
        )
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            command = json.loads(raw_line)
            if command.get("command") == "shutdown":
                _emit(events, "STOPPED", worker_id=worker_id, gpu_id=gpu_id)
                return
            if command.get("command") != "run":
                raise ValueError(f"Unknown worker command: {command}")

            suite = str(command["suite"])
            task_id = int(command["task_id"])
            attempt = int(command["attempt"])
            _emit(
                events,
                "STARTED",
                worker_id=worker_id,
                gpu_id=gpu_id,
                suite=suite,
                task_id=task_id,
                attempt=attempt,
            )
            try:
                results, output_file = evaluate_task(
                    cfg,
                    runtime,
                    task_suite_name=suite,
                    task_id=task_id,
                    gpu_id=gpu_id,
                )
            except BaseException as exc:
                _emit(
                    events,
                    "FAILED",
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    suite=suite,
                    task_id=task_id,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                raise
            _emit(
                events,
                "SUCCEEDED",
                worker_id=worker_id,
                gpu_id=gpu_id,
                suite=suite,
                task_id=task_id,
                attempt=attempt,
                output_file=str(output_file),
                duration=float(results["duration"]),
                timings=results.get("timings", {}),
            )
    except BaseException as exc:
        _emit(
            events,
            "WORKER_FATAL",
            worker_id=worker_id,
            gpu_id=gpu_id,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        events.close()


if __name__ == "__main__":
    main()
