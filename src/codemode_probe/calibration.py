from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from codemode_probe.models import RankedCandidate, ScoreFailureReason, StructuredAnswer
from codemode_probe.scoring import score_answer


class CalibrationCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    details: dict[str, object]


def run_scoring_calibration_checks() -> list[CalibrationCheckResult]:
    return [
        _schema_invalid_check(),
        _task_id_mismatch_check(),
        _partial_overlap_check(),
        _hallucinated_ids_check(),
        _duplicate_ids_check(),
        _degraded_ranking_check(),
    ]


def _schema_invalid_check() -> CalibrationCheckResult:
    oracle = _answer("task-1", ["a"])
    score = score_answer({"task_id": "task-1", "candidates": [{"id": "a"}]}, oracle)
    return _check(
        "calibration_schema_invalid",
        passed=(
            score.schema_valid is False
            and score.failure_reason == ScoreFailureReason.SCHEMA_INVALID
            and score.top_k_overlap == 0.0
            and score.ndcg_at_k == 0.0
        ),
        details=score.model_dump(mode="json"),
    )


def _task_id_mismatch_check() -> CalibrationCheckResult:
    score = score_answer(_answer("other-task", ["a"]), _answer("task-1", ["a"]))
    return _check(
        "calibration_task_id_mismatch",
        passed=(
            score.schema_valid is True
            and score.failure_reason == ScoreFailureReason.TASK_ID_MISMATCH
            and score.top_k_overlap == 0.0
            and score.ndcg_at_k == 0.0
        ),
        details=score.model_dump(mode="json"),
    )


def _partial_overlap_check() -> CalibrationCheckResult:
    score = score_answer(_answer("task-1", ["a", "x", "b"]), _answer("task-1", ["a", "b", "c"]))
    return _check(
        "calibration_partial_overlap",
        passed=(
            score.schema_valid is True
            and score.failure_reason is None
            and round(score.top_k_overlap, 6) == round(2 / 3, 6)
            and round(score.precision_at_k, 6) == round(2 / 3, 6)
            and round(score.recall_at_k, 6) == round(2 / 3, 6)
        ),
        details=score.model_dump(mode="json"),
    )


def _hallucinated_ids_check() -> CalibrationCheckResult:
    score = score_answer(_answer("task-1", ["x", "y", "z"]), _answer("task-1", ["a", "b", "c"]))
    return _check(
        "calibration_hallucinated_ids",
        passed=(
            score.schema_valid is True
            and score.failure_reason is None
            and score.top_k_overlap == 0.0
            and score.ndcg_at_k == 0.0
        ),
        details=score.model_dump(mode="json"),
    )


def _duplicate_ids_check() -> CalibrationCheckResult:
    score = score_answer(_answer("task-1", ["a", "a", "b"]), _answer("task-1", ["a", "b", "c"]))
    return _check(
        "calibration_duplicate_ids",
        passed=(
            score.schema_valid is True
            and score.failure_reason is None
            and round(score.top_k_overlap, 6) == round(2 / 3, 6)
            and round(score.precision_at_k, 6) == round(2 / 3, 6)
        ),
        details=score.model_dump(mode="json"),
    )


def _degraded_ranking_check() -> CalibrationCheckResult:
    oracle = _answer("task-1", ["a", "b", "c", "d"])
    perfect = score_answer(_answer("task-1", ["a", "b", "c", "d"]), oracle)
    degraded = score_answer(_answer("task-1", ["d", "c", "b", "a"]), oracle)
    return _check(
        "calibration_degraded_ranking",
        passed=(
            degraded.schema_valid is True
            and degraded.failure_reason is None
            and degraded.top_k_overlap == 1.0
            and 0.0 < degraded.ndcg_at_k < perfect.ndcg_at_k
        ),
        details={
            "perfect": perfect.model_dump(mode="json"),
            "degraded": degraded.model_dump(mode="json"),
        },
    )


def _check(name: str, *, passed: bool, details: dict[str, object]) -> CalibrationCheckResult:
    return CalibrationCheckResult(name=name, passed=passed, details=details)


def _answer(task_id: str, candidate_ids: list[str]) -> StructuredAnswer:
    return StructuredAnswer(
        task_id=task_id,
        candidates=[
            RankedCandidate(id=candidate_id, score=1.0)
            for candidate_id in candidate_ids
        ],
    )
