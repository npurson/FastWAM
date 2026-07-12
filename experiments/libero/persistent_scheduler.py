"""Persistent subprocess scheduler for LIBERO evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int

    @property
    def key(self) -> str:
        return f"{self.suite},{self.task_id}"


@dataclass
class WorkerHandle:
    worker_id: str
    gpu_id: int
    slot: int
    process: subprocess.Popen[str]
    log_handle: Any
    generation: int
    launched_at: float
    ready: bool = False
    assigned: TaskSpec | None = None
    assigned_at: float | None = None
    attempt: int | None = None
    restart_count: int = 0
    disabled: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _load_tasks(task_file: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(task_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid task row at {task_file}:{line_number}: {raw_line!r}")
        task = TaskSpec(parts[0].strip(), int(parts[1]))
        if task.key in seen:
            raise ValueError(f"Duplicate task in {task_file}: {task.key}")
        seen.add(task.key)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"Task file is empty: {task_file}")
    return tasks


def _valid_result(output_dir: Path, task: TaskSpec, num_trials: int) -> Path | None:
    for candidate in sorted((output_dir / task.suite).glob(f"gpu*_task{task.task_id}_results.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if (
                str(payload.get("task_suite")) == task.suite
                and int(payload.get("task_id")) == task.task_id
                and int(payload.get("total_episodes")) == int(num_trials)
            ):
                return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


class PersistentScheduler:
    def __init__(
        self,
        *,
        task_file: Path,
        task_choice: str,
        ckpt: str,
        gpu_ids: list[int],
        num_trials: int,
        workers_per_gpu: int,
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
        self.task_file = task_file.resolve()
        self.task_choice = task_choice
        self.ckpt = ckpt
        self.gpu_ids = gpu_ids
        self.num_trials = int(num_trials)
        self.workers_per_gpu = int(workers_per_gpu)
        self.output_dir = output_dir.resolve()
        self.extra_overrides = list(extra_overrides)
        self.resume = bool(resume)
        self.max_task_retries = int(max_task_retries)
        self.task_timeout_seconds = int(task_timeout_seconds)
        self.worker_start_timeout_seconds = int(worker_start_timeout_seconds)
        self.max_worker_restarts = int(max_worker_restarts)
        self.worker_launch_interval_seconds = float(worker_launch_interval_seconds)
        self.resolved_config_sha256 = str(resolved_config_sha256)
        if not self.gpu_ids:
            raise ValueError("Persistent scheduler requires at least one GPU.")
        if self.workers_per_gpu <= 0:
            raise ValueError("MULTIRUN.max_tasks_per_gpu must be positive.")
        if self.max_task_retries < 0:
            raise ValueError("MULTIRUN.max_task_retries must be non-negative.")
        if self.task_timeout_seconds <= 0 or self.worker_start_timeout_seconds <= 0:
            raise ValueError("Worker and task timeouts must be positive.")
        self.tasks = _load_tasks(self.task_file)
        self.tasks_by_key = {task.key: task for task in self.tasks}
        self.pending: deque[TaskSpec] = deque()
        self.completed: set[str] = set()
        self.final_failures: dict[str, dict[str, Any]] = {}
        self.attempts: dict[str, int] = {}
        self.timing_totals: dict[str, float] = {}
        self.task_duration_total = 0.0
        self.workers: dict[str, WorkerHandle] = {}
        self.event_queue: queue.Queue[tuple[str, int, int, dict[str, Any]]] = queue.Queue()
        self.stop_requested = False
        self.started_at = time.time()
        self.last_status_at = 0.0
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.events_path = self.output_dir / "worker_events.jsonl"
        self.failures_path = self.output_dir / "failed_tasks.jsonl"
        self.worker_log_dir = self.output_dir / "worker_logs"
        self.worker_log_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_manifest_and_resume()

    def _fingerprint_payload(self) -> dict[str, Any]:
        checkpoint = Path(self.ckpt).expanduser().resolve()
        checkpoint_stat = checkpoint.stat() if checkpoint.exists() else None
        return {
            "task_file_sha256": _sha256_file(self.task_file),
            "task_choice": self.task_choice,
            "checkpoint": str(checkpoint),
            "checkpoint_size": None if checkpoint_stat is None else checkpoint_stat.st_size,
            "checkpoint_mtime_ns": None if checkpoint_stat is None else checkpoint_stat.st_mtime_ns,
            "num_trials": self.num_trials,
            "extra_overrides": self.extra_overrides,
            "resolved_config_sha256": self.resolved_config_sha256,
        }

    def _prepare_manifest_and_resume(self) -> None:
        fingerprint_payload = self._fingerprint_payload()
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        old_manifest = None
        if self.manifest_path.exists():
            old_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not self.resume:
                raise FileExistsError(
                    f"Run manifest already exists and MULTIRUN.resume=false: {self.manifest_path}"
                )
        elif any(self.output_dir.glob("*/gpu*_task*_results.json")):
            raise ValueError(
                "Result files exist without run_manifest.json, so their evaluation provenance "
                "cannot be verified for resume. Use a fresh output directory."
            )
            if old_manifest.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Refusing to resume with a different task/checkpoint/evaluation configuration. "
                    f"Existing manifest: {self.manifest_path}"
                )

        for task in self.tasks:
            existing_result = _valid_result(self.output_dir, task, self.num_trials) if self.resume else None
            if existing_result is not None:
                self.completed.add(task.key)
                try:
                    self._accumulate_result_payload(
                        json.loads(existing_result.read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            else:
                self.pending.append(task)

        if old_manifest is not None and len(self.completed) == len(self.tasks):
            self.manifest = old_manifest
            return

        self.manifest: dict[str, Any] = {
            "version": 1,
            "backend": "persistent_subprocess_workers",
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "task_count": len(self.tasks),
            "gpu_ids": self.gpu_ids,
            "workers_per_gpu": self.workers_per_gpu,
            "worker_count": len(self.gpu_ids) * self.workers_per_gpu,
            "max_task_retries": self.max_task_retries,
            "task_timeout_seconds": self.task_timeout_seconds,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "status": "completed" if len(self.completed) == len(self.tasks) else "running",
            "completed": len(self.completed),
            "pending": len(self.pending),
            "failed": 0,
            "model_load_events": [],
        }
        _atomic_json(self.manifest_path, self.manifest)

    def _accumulate_result_payload(self, payload: dict[str, Any]) -> None:
        self.task_duration_total += float(payload.get("duration", 0.0))
        for name, value in payload.get("timings", {}).items():
            self.timing_totals[str(name)] = self.timing_totals.get(str(name), 0.0) + float(value)

    def _record_event(self, event: dict[str, Any]) -> None:
        event = {"timestamp": time.time(), **event}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _reader(self, worker_id: str, pid: int, generation: int, read_fd: int) -> None:
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8", buffering=1) as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        event = {"event": "PROTOCOL_ERROR", "error": str(exc), "raw": line}
                    self.event_queue.put((worker_id, pid, generation, event))
        finally:
            self.event_queue.put((worker_id, pid, generation, {"event": "EVENT_EOF"}))

    def _spawn_worker(self, worker_id: str, gpu_id: int, slot: int, restart_count: int) -> WorkerHandle:
        read_fd, write_fd = os.pipe()
        generation = restart_count
        log_path = self.worker_log_dir / f"{worker_id}.generation{generation}.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "LIBERO_EVENT_FD": str(write_fd),
                "LIBERO_WORKER_ID": worker_id,
                "LIBERO_GPU_ID": str(gpu_id),
                "LIBERO_OUTPUT_DIR": str(self.output_dir),
                "LIBERO_NUM_TRIALS": str(self.num_trials),
            }
        )
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        worker_script = os.environ.get(
            "LIBERO_WORKER_SCRIPT", "experiments/libero/eval_libero_worker.py"
        )
        command = [
            sys.executable,
            worker_script,
            f"task={self.task_choice}",
            f"ckpt={self.ckpt}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.output_dir={self.output_dir}",
            f"EVALUATION.num_trials={self.num_trials}",
            *self.extra_overrides,
        ]
        try:
            process = subprocess.Popen(
                command,
            cwd=PROJECT_ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                pass_fds=(write_fd,),
                start_new_session=True,
            )
        finally:
            os.close(write_fd)
        handle = WorkerHandle(
            worker_id=worker_id,
            gpu_id=gpu_id,
            slot=slot,
            process=process,
            log_handle=log_handle,
            generation=generation,
            launched_at=time.monotonic(),
            restart_count=restart_count,
        )
        threading.Thread(
            target=self._reader,
            args=(worker_id, process.pid, generation, read_fd),
            daemon=True,
        ).start()
        self._record_event(
            {"event": "WORKER_SPAWNED", "worker_id": worker_id, "gpu_id": gpu_id, "pid": process.pid}
        )
        return handle

    def _start_initial_workers(self) -> None:
        for gpu_id in self.gpu_ids:
            for slot in range(self.workers_per_gpu):
                worker_id = f"gpu{gpu_id}_slot{slot}"
                self.workers[worker_id] = self._spawn_worker(worker_id, gpu_id, slot, 0)
                if self.worker_launch_interval_seconds > 0:
                    time.sleep(self.worker_launch_interval_seconds)

    def _terminate_worker(self, worker: WorkerHandle, *, force: bool = False) -> None:
        process = worker.process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10 if not force else 2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if process.stdin is not None:
            process.stdin.close()
        worker.log_handle.close()

    def _handle_task_failure(self, worker: WorkerHandle, reason: str, traceback_text: str = "") -> None:
        task = worker.assigned
        attempt = worker.attempt
        worker.assigned = None
        worker.assigned_at = None
        worker.attempt = None
        worker.ready = False
        if task is None or attempt is None:
            return
        # The worker may have committed its atomic result and then lost the
        # event pipe. Treat a valid result as authoritative to avoid rerunning
        # an already completed benchmark task.
        recovered_result = _valid_result(self.output_dir, task, self.num_trials)
        if recovered_result is not None:
            self.completed.add(task.key)
            try:
                self._accumulate_result_payload(
                    json.loads(recovered_result.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            print(
                f"Recovered completed task {task.key} from its result JSON after worker failure: {reason}",
                flush=True,
            )
            return
        failure = {
            "suite": task.suite,
            "task_id": task.task_id,
            "attempt": attempt,
            "worker_id": worker.worker_id,
            "gpu_id": worker.gpu_id,
            "reason": reason,
            "traceback": traceback_text,
            "timestamp": time.time(),
        }
        with self.failures_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        if attempt <= self.max_task_retries:
            self.pending.appendleft(task)
            print(f"Task {task.key} failed on attempt {attempt}; requeued: {reason}", flush=True)
        else:
            self.final_failures[task.key] = failure
            print(f"Task {task.key} exhausted retries: {reason}", flush=True)

    def _handle_event(self, worker_id: str, pid: int, generation: int, event: dict[str, Any]) -> None:
        worker = self.workers.get(worker_id)
        if worker is None or worker.process.pid != pid or worker.generation != generation:
            return
        self._record_event(event)
        event_type = event.get("event")
        if event_type == "READY":
            worker.ready = True
            # Count only consecutive startup failures. Once a replacement has
            # loaded successfully, later task-level failures get a fresh budget.
            worker.restart_count = 0
            self.manifest["model_load_events"].append(
                {
                    "worker_id": worker_id,
                    "gpu_id": worker.gpu_id,
                    "generation": generation,
                    "seconds": float(event.get("model_load_seconds", 0.0)),
                }
            )
        elif event_type == "STARTED":
            worker.assigned_at = time.monotonic()
        elif event_type == "SUCCEEDED":
            task = worker.assigned
            if task is None or event.get("suite") != task.suite or int(event.get("task_id")) != task.task_id:
                self._handle_task_failure(worker, f"Protocol task mismatch: {event}")
                self._terminate_worker(worker, force=True)
                return
            if _valid_result(self.output_dir, task, self.num_trials) is None:
                self._handle_task_failure(worker, "Worker reported success without a valid result JSON")
                self._terminate_worker(worker, force=True)
                return
            self.completed.add(task.key)
            self.task_duration_total += float(event.get("duration", 0.0))
            for name, value in event.get("timings", {}).items():
                self.timing_totals[str(name)] = self.timing_totals.get(str(name), 0.0) + float(value)
            worker.assigned = None
            worker.assigned_at = None
            worker.attempt = None
            worker.ready = True
        elif event_type == "FAILED":
            self._handle_task_failure(worker, str(event.get("error", "worker task failure")), str(event.get("traceback", "")))
        elif event_type in {"WORKER_FATAL", "PROTOCOL_ERROR"}:
            if worker.assigned is not None:
                self._handle_task_failure(worker, str(event.get("error", event_type)), str(event.get("traceback", "")))
            worker.ready = False

    def _check_workers(self) -> None:
        now = time.monotonic()
        for worker_id, worker in list(self.workers.items()):
            return_code = worker.process.poll()
            if return_code is not None:
                if worker.assigned is not None:
                    self._handle_task_failure(worker, f"Worker exited with return code {return_code}")
                worker.log_handle.close()
                if worker.process.stdin is not None and not worker.process.stdin.closed:
                    worker.process.stdin.close()
                if worker.restart_count >= self.max_worker_restarts:
                    worker.disabled = True
                    continue
                replacement = self._spawn_worker(
                    worker_id, worker.gpu_id, worker.slot, worker.restart_count + 1
                )
                self.workers[worker_id] = replacement
                continue
            if not worker.ready and worker.assigned is None:
                if now - worker.launched_at > self.worker_start_timeout_seconds:
                    print(f"Worker {worker_id} startup timed out; restarting", flush=True)
                    self._terminate_worker(worker, force=True)
                continue
            if worker.assigned is not None and worker.assigned_at is not None:
                if now - worker.assigned_at > self.task_timeout_seconds:
                    self._handle_task_failure(worker, f"Task exceeded {self.task_timeout_seconds}s timeout")
                    self._terminate_worker(worker, force=True)

    def _dispatch(self) -> None:
        for worker in self.workers.values():
            if not self.pending:
                return
            if worker.disabled or not worker.ready or worker.assigned is not None or worker.process.poll() is not None:
                continue
            task = self.pending.popleft()
            attempt = self.attempts.get(task.key, 0) + 1
            self.attempts[task.key] = attempt
            command = {"command": "run", "suite": task.suite, "task_id": task.task_id, "attempt": attempt}
            try:
                assert worker.process.stdin is not None
                worker.process.stdin.write(json.dumps(command) + "\n")
                worker.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.pending.appendleft(task)
                worker.ready = False
                self._terminate_worker(worker, force=True)
                print(f"Failed to dispatch {task.key} to {worker.worker_id}: {exc}", flush=True)
                continue
            worker.assigned = task
            worker.assigned_at = time.monotonic()
            worker.attempt = attempt
            worker.ready = False

    def _write_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_status_at < 30:
            return
        running = sum(worker.assigned is not None for worker in self.workers.values())
        ready = sum(worker.ready and worker.assigned is None for worker in self.workers.values())
        self.manifest.update(
            {
                "completed": len(self.completed),
                "pending": len(self.pending),
                "running": running,
                "ready_workers": ready,
                "failed": len(self.final_failures),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_duration_total_seconds": self.task_duration_total,
                "timing_totals_seconds": self.timing_totals,
                "throughput_tasks_per_hour": (
                    len(self.completed) * 3600.0 / max(time.time() - self.started_at, 1e-9)
                ),
            }
        )
        _atomic_json(self.manifest_path, self.manifest)
        print(
            f"Scheduling status: completed={len(self.completed)}/{len(self.tasks)} "
            f"running={running} pending={len(self.pending)} failed={len(self.final_failures)} "
            f"ready_workers={ready}",
            flush=True,
        )
        self.last_status_at = now

    def request_stop(self, signum, _frame) -> None:
        print(f"Received signal {signum}; stopping persistent scheduler", flush=True)
        self.stop_requested = True

    def _shutdown_workers(self) -> None:
        for worker in self.workers.values():
            if worker.process.poll() is None and worker.process.stdin is not None:
                try:
                    worker.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    worker.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
        deadline = time.monotonic() + 20
        for worker in self.workers.values():
            if worker.process.poll() is None:
                timeout = max(0.0, deadline - time.monotonic())
                try:
                    worker.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._terminate_worker(worker, force=True)
                    continue
            if not worker.log_handle.closed:
                worker.log_handle.close()
            if worker.process.stdin is not None and not worker.process.stdin.closed:
                worker.process.stdin.close()

    def run(self) -> None:
        if len(self.completed) == len(self.tasks):
            self.manifest.update(
                {
                    "status": "completed",
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "wall_seconds": time.time() - self.started_at,
                }
            )
            self._write_status(force=True)
            _atomic_json(self.manifest_path, self.manifest)
            return
        previous_handlers = {
            signal.SIGINT: signal.signal(signal.SIGINT, self.request_stop),
            signal.SIGTERM: signal.signal(signal.SIGTERM, self.request_stop),
        }
        caught_error: BaseException | None = None
        caught_traceback = None
        try:
            self._start_initial_workers()
            while not self.stop_requested:
                try:
                    worker_id, pid, generation, event = self.event_queue.get(timeout=1.0)
                    self._handle_event(worker_id, pid, generation, event)
                    while True:
                        worker_id, pid, generation, event = self.event_queue.get_nowait()
                        self._handle_event(worker_id, pid, generation, event)
                except queue.Empty:
                    pass
                self._check_workers()
                self._dispatch()
                self._write_status()

                accounted = len(self.completed) + len(self.final_failures)
                running = any(worker.assigned is not None for worker in self.workers.values())
                if accounted == len(self.tasks) and not self.pending and not running:
                    break
                available = any(not worker.disabled for worker in self.workers.values())
                if not available:
                    raise RuntimeError("All persistent workers exceeded the restart limit.")
        except BaseException as exc:
            caught_error = exc
            caught_traceback = exc.__traceback__
        finally:
            self._shutdown_workers()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

        self.manifest["status"] = (
            "interrupted"
            if self.stop_requested
            else ("failed" if caught_error is not None or self.final_failures else "completed")
        )
        self.manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manifest["wall_seconds"] = time.time() - self.started_at
        self._write_status(force=True)
        _atomic_json(self.manifest_path, self.manifest)
        if caught_error is not None:
            raise caught_error.with_traceback(caught_traceback)
        if self.stop_requested:
            raise KeyboardInterrupt("LIBERO evaluation interrupted")
        if self.final_failures:
            raise RuntimeError(
                f"{len(self.final_failures)} LIBERO tasks failed after retries; see {self.failures_path}"
            )


def visible_gpu_ids(num_gpus: int) -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return list(range(int(num_gpus)))
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != int(num_gpus):
        raise ValueError(
            f"MULTIRUN.num_gpus={num_gpus}, but CUDA_VISIBLE_DEVICES contains {len(values)} entries: {raw}"
        )
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError("Persistent LIBERO scheduler currently requires numeric CUDA device IDs.") from exc
