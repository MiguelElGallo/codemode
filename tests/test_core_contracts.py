from __future__ import annotations

import pytest

from codemode_probe.models import (
    Candidate,
    RankedCandidate,
    ScoreFailureReason,
    StructuredAnswer,
    WorkloadConfig,
)
from codemode_probe.oracle import candidate_score, rank_candidates
from codemode_probe.scoring import score_answer
from codemode_probe.workload import candidates_by_shard, generate_candidates


def make_candidate(candidate_id: str, **overrides: object) -> Candidate:
    values = {
        "id": candidate_id,
        "shard_id": 0,
        "title": f"candidate {candidate_id}",
        "category": "tests",
        "age_days": 10,
        "approvals": 2,
        "failing_checks": 0,
        "reactions": 10,
        "changed_files": 20,
        "relevance": 0.5,
    }
    values.update(overrides)
    return Candidate(**values)


def answer(task_id: str, ids: list[str]) -> StructuredAnswer:
    return StructuredAnswer(
        task_id=task_id,
        candidates=[RankedCandidate(id=candidate_id, score=1.0) for candidate_id in ids],
    )


def test_generate_candidates_is_deterministic_for_same_workload() -> None:
    config = WorkloadConfig(
        seed=123,
        shard_count=3,
        candidates_per_shard=4,
        payload_bytes=16,
        relevant_fraction=0.25,
    )

    first = generate_candidates(config)
    second = generate_candidates(config)
    different_seed = generate_candidates(config.model_copy(update={"seed": 124}))

    assert [candidate.model_dump() for candidate in first] == [
        candidate.model_dump() for candidate in second
    ]
    assert [candidate.model_dump() for candidate in first] != [
        candidate.model_dump() for candidate in different_seed
    ]


def test_generate_candidates_respects_shard_and_payload_sizing() -> None:
    config = WorkloadConfig(
        seed=9,
        shard_count=4,
        candidates_per_shard=3,
        payload_bytes=32,
        relevant_fraction=0.5,
    )

    candidates = generate_candidates(config)
    shards = candidates_by_shard(candidates)

    assert len(candidates) == 12
    assert set(shards) == {0, 1, 2, 3}
    assert {shard_id: len(items) for shard_id, items in shards.items()} == {
        0: 3,
        1: 3,
        2: 3,
        3: 3,
    }
    assert {len(candidate.payload) for candidate in candidates} == {32}
    assert generate_candidates(config.model_copy(update={"payload_bytes": 0}))[0].payload == ""


def test_generate_candidates_allows_zero_relevance_workload() -> None:
    config = WorkloadConfig(seed=9, shard_count=2, candidates_per_shard=5, relevant_fraction=0.0)

    candidates = generate_candidates(config)

    assert all(candidate.relevance <= 0.5 for candidate in candidates)


def test_oracle_excludes_drafts_and_bots_and_breaks_score_ties_by_id() -> None:
    included_b = make_candidate("cand-b")
    included_a = make_candidate("cand-a")
    draft = make_candidate("cand-draft", approvals=4, relevance=1.0, is_draft=True)
    bot = make_candidate("cand-bot", approvals=4, relevance=1.0, is_bot_authored=True)

    oracle = rank_candidates("task-1", [included_b, draft, bot, included_a], top_k=10)

    assert [candidate.id for candidate in oracle.candidates] == ["cand-a", "cand-b"]
    assert candidate_score(draft) == (0.0, {"excluded": 1.0})
    assert candidate_score(bot) == (0.0, {"excluded": 1.0})


def test_score_answer_rejects_invalid_schema() -> None:
    oracle = answer("task-1", ["expected"])

    result = score_answer({"task_id": "task-1", "candidates": [{"id": "expected"}]}, oracle)

    assert result.schema_valid is False
    assert result.failure_reason == ScoreFailureReason.SCHEMA_INVALID
    assert result.top_k_overlap == 0.0
    assert result.ndcg_at_k == 0.0


def test_score_answer_reports_overlap_precision_and_recall() -> None:
    oracle = answer("task-1", ["a", "b", "c", "d"])
    actual = answer("task-1", ["d", "x", "b"])

    result = score_answer(actual, oracle)

    assert result.schema_valid is True
    assert result.top_k_overlap == 0.5
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.recall_at_k == 0.5


def test_score_answer_rejects_task_id_mismatch() -> None:
    oracle = answer("task-1", ["a", "b"])
    actual = answer("other-task", ["a", "b"])

    result = score_answer(actual, oracle)

    assert result.schema_valid is True
    assert result.failure_reason == ScoreFailureReason.TASK_ID_MISMATCH
    assert result.top_k_overlap == 0.0
    assert result.ndcg_at_k == 0.0


def test_score_answer_ndcg_rewards_expected_ordering() -> None:
    oracle = answer("task-1", ["a", "b", "c", "d"])

    perfect = score_answer(answer("task-1", ["a", "b", "c", "d"]), oracle)
    reversed_order = score_answer(answer("task-1", ["d", "c", "b", "a"]), oracle)
    irrelevant = score_answer(answer("task-1", ["w", "x", "y", "z"]), oracle)

    assert perfect.ndcg_at_k == pytest.approx(1.0)
    assert 0.0 < reversed_order.ndcg_at_k < perfect.ndcg_at_k
    assert irrelevant.ndcg_at_k == 0.0
