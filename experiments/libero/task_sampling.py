import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskCandidate:
    suite: str
    task_id: int
    name: str
    category: str | None = None
    difficulty_level: int | None = None


def load_classification(
    classification_path: Path,
    candidates: Iterable[TaskCandidate],
) -> tuple[list[TaskCandidate], str]:
    raw_bytes = classification_path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected classification metadata to be a mapping, got {type(payload)}: "
            f"{classification_path}"
        )

    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for suite, rows in payload.items():
        if not isinstance(rows, list):
            raise ValueError(f"Classification rows for {suite} must be a list.")
        for expected_id, row in enumerate(rows, start=1):
            if int(row.get("id", -1)) != expected_id:
                raise ValueError(
                    f"Classification IDs for {suite} must be contiguous and 1-based; "
                    f"expected {expected_id}, got {row.get('id')}."
                )
            rows_by_key[(str(suite), expected_id - 1)] = row

    classified: list[TaskCandidate] = []
    for candidate in candidates:
        key = (candidate.suite, candidate.task_id)
        row = rows_by_key.get(key)
        if row is None:
            raise ValueError(f"Missing classification metadata for task {key}.")
        if str(row.get("name")) != candidate.name:
            raise ValueError(
                "Classification metadata does not match the installed benchmark task order: "
                f"{key} metadata={row.get('name')!r} benchmark={candidate.name!r}."
            )
        difficulty = row.get("difficulty_level")
        classified.append(
            TaskCandidate(
                suite=candidate.suite,
                task_id=candidate.task_id,
                name=candidate.name,
                category=str(row["category"]),
                difficulty_level=None if difficulty is None else int(difficulty),
            )
        )

    return classified, hashlib.sha256(raw_bytes).hexdigest()


def stratified_sample(
    candidates: Iterable[TaskCandidate],
    *,
    ratio: float,
    seed: int,
) -> list[TaskCandidate]:
    candidates = list(candidates)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {ratio}")
    if ratio == 1.0:
        return candidates
    if any(x.category is None for x in candidates):
        raise ValueError("Stratified sampling requires category metadata for every task.")

    groups: dict[tuple[str, str, int | None], list[TaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate.suite, str(candidate.category), candidate.difficulty_level)].append(candidate)

    target_total = int(math.floor(len(candidates) * ratio + 0.5))
    suite_sizes = Counter(x.suite for x in candidates)
    suite_allocations = {suite: int(math.floor(size * ratio)) for suite, size in suite_sizes.items()}
    suite_remainders = [
        (size * ratio - suite_allocations[suite], suite)
        for suite, size in suite_sizes.items()
    ]
    suite_remaining = target_total - sum(suite_allocations.values())
    for _, suite in sorted(suite_remainders, key=lambda x: (-x[0], x[1]))[:suite_remaining]:
        suite_allocations[suite] += 1

    allocations: dict[tuple[str, str, int | None], int] = {}
    for suite, suite_target in suite_allocations.items():
        suite_groups = [(key, values) for key, values in groups.items() if key[0] == suite]
        remainders: list[tuple[float, str, tuple[str, str, int | None]]] = []
        for key, values in suite_groups:
            exact = len(values) * ratio
            # Preserve coverage of rare classification strata whenever the
            # suite-level sample budget is large enough (as it is at 20%).
            base = max(1, int(math.floor(exact)))
            allocations[key] = base
            remainders.append((exact - base, repr(key), key))

        remaining = suite_target - sum(allocations[key] for key, _ in suite_groups)
        if remaining < 0 or remaining > len(remainders):
            raise RuntimeError(
                "Invalid hierarchical largest-remainder allocation: "
                f"suite={suite} target={suite_target} remaining={remaining}"
            )
        for _, _, key in sorted(remainders, key=lambda x: (-x[0], x[1]))[:remaining]:
            allocations[key] += 1

    rng = random.Random(int(seed))
    sampled: list[TaskCandidate] = []
    for key in sorted(groups, key=repr):
        values = sorted(groups[key], key=lambda x: x.task_id)
        sampled.extend(rng.sample(values, allocations[key]))

    suite_order = {suite: idx for idx, suite in enumerate(dict.fromkeys(x.suite for x in candidates))}
    sampled.sort(key=lambda x: (suite_order[x.suite], x.task_id))
    if len(sampled) != target_total:
        raise RuntimeError(f"Sample size mismatch: got {len(sampled)}, expected {target_total}")
    return sampled


def build_sampling_manifest(
    *,
    all_candidates: Iterable[TaskCandidate],
    sampled_candidates: Iterable[TaskCandidate],
    ratio: float,
    seed: int,
    classification_path: Path,
    classification_sha256: str,
) -> dict[str, Any]:
    all_candidates = list(all_candidates)
    sampled_candidates = list(sampled_candidates)

    def _counts(values: list[TaskCandidate], fields: tuple[str, ...]) -> dict[str, int]:
        counts = Counter(tuple(getattr(value, field) for field in fields) for value in values)
        return {" | ".join(map(str, key)): count for key, count in sorted(counts.items(), key=lambda x: repr(x[0]))}

    return {
        "protocol": "sampled_libero_plus",
        "strategy": "hierarchical_largest_remainder_by_suite_then_category_difficulty",
        "sample_ratio": float(ratio),
        "sample_seed": int(seed),
        "full_task_count": len(all_candidates),
        "sampled_task_count": len(sampled_candidates),
        "classification_path": str(classification_path),
        "classification_sha256": classification_sha256,
        "full_counts_by_suite": _counts(all_candidates, ("suite",)),
        "sampled_counts_by_suite": _counts(sampled_candidates, ("suite",)),
        "full_counts_by_stratum": _counts(
            all_candidates, ("suite", "category", "difficulty_level")
        ),
        "sampled_counts_by_stratum": _counts(
            sampled_candidates, ("suite", "category", "difficulty_level")
        ),
    }
