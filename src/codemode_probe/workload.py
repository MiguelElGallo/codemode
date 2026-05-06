from __future__ import annotations

import random
import string

from codemode_probe.models import Candidate, ProbeTask, TaskFamily, ToolShape, WorkloadConfig


_CATEGORIES = ("api", "docs", "infra", "runtime", "tests")


def make_probe_task(
    task_id: str,
    *,
    seed: int = 1,
    task_family: TaskFamily = TaskFamily.SCALAR_LARGE_FANOUT,
    tool_shape: ToolShape = ToolShape.SCALAR,
    shard_count: int = 5,
    candidates_per_shard: int = 20,
    payload_bytes: int = 256,
    relevant_fraction: float = 0.2,
    top_k: int = 5,
) -> ProbeTask:
    workload = WorkloadConfig(
        seed=seed,
        task_family=task_family,
        tool_shape=tool_shape,
        shard_count=shard_count,
        candidates_per_shard=candidates_per_shard,
        payload_bytes=payload_bytes,
        relevant_fraction=relevant_fraction,
        top_k=top_k,
    )
    prompt = (
        "Rank the top candidates most ready to merge. Exclude drafts and "
        "bot-authored candidates. Consider approvals, CI status, reactions, "
        "recency, changed-file count, and relevance. Return structured JSON."
    )
    return ProbeTask(id=task_id, prompt=prompt, workload=workload)


def generate_candidates(config: WorkloadConfig) -> list[Candidate]:
    rng = random.Random(config.seed)
    relevant_cutoff = round(config.candidate_count * config.relevant_fraction)
    relevant_ids = set(rng.sample(range(config.candidate_count), k=relevant_cutoff))
    candidates: list[Candidate] = []

    for index in range(config.candidate_count):
        shard_id = index // config.candidates_per_shard
        is_relevant = index in relevant_ids
        relevance = rng.uniform(0.65, 1.0) if is_relevant else rng.uniform(0.0, 0.5)
        payload = _make_payload(rng, config.payload_bytes)
        candidates.append(
            Candidate(
                id=f"cand-{index:04d}",
                shard_id=shard_id,
                title=f"{rng.choice(_CATEGORIES)} candidate {index}",
                category=rng.choice(_CATEGORIES),
                age_days=rng.randint(0, 45),
                approvals=rng.randint(0, 4) + (1 if is_relevant else 0),
                failing_checks=max(0, rng.randint(0, 3) - (1 if is_relevant else 0)),
                reactions=rng.randint(0, 30) + (10 if is_relevant else 0),
                changed_files=rng.randint(1, 80),
                is_draft=rng.random() < (0.03 if is_relevant else 0.12),
                is_bot_authored=rng.random() < (0.02 if is_relevant else 0.10),
                relevance=round(relevance, 4),
                payload=payload,
            )
        )

    return candidates


def candidates_by_shard(candidates: list[Candidate]) -> dict[int, list[Candidate]]:
    shards: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        shards.setdefault(candidate.shard_id, []).append(candidate)
    return shards


def _make_payload(rng: random.Random, size: int) -> str:
    if size <= 0:
        return ""
    alphabet = string.ascii_letters + string.digits + " "
    return "".join(rng.choice(alphabet) for _ in range(size))
