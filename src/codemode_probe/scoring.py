from __future__ import annotations

import math

from pydantic import ValidationError

from codemode_probe.models import ScoreFailureReason, ScoreResult, StructuredAnswer


def score_answer(
    answer: StructuredAnswer | dict,
    oracle: StructuredAnswer,
    *,
    timed_out: bool = False,
) -> ScoreResult:
    parsed = _parse_answer(answer)
    if parsed is None:
        return ScoreResult(
            schema_valid=False,
            timed_out=timed_out,
            top_k_overlap=0.0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            failure_reason=ScoreFailureReason.SCHEMA_INVALID,
        )

    if parsed.task_id != oracle.task_id:
        return ScoreResult(
            schema_valid=True,
            timed_out=timed_out,
            top_k_overlap=0.0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            failure_reason=ScoreFailureReason.TASK_ID_MISMATCH,
        )

    expected_ids = [candidate.id for candidate in oracle.candidates]
    actual_ids = [candidate.id for candidate in parsed.candidates[: len(expected_ids)]]
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    overlap = len(expected_set & actual_set)
    k = max(1, len(expected_ids))

    return ScoreResult(
        schema_valid=True,
        timed_out=timed_out,
        top_k_overlap=overlap / k,
        precision_at_k=overlap / max(1, len(actual_ids)),
        recall_at_k=overlap / k,
        ndcg_at_k=_ndcg(actual_ids, expected_ids),
        failure_reason=ScoreFailureReason.TIMEOUT if timed_out else None,
    )


def _parse_answer(answer: StructuredAnswer | dict) -> StructuredAnswer | None:
    if isinstance(answer, StructuredAnswer):
        return answer
    try:
        return StructuredAnswer.model_validate(answer)
    except ValidationError:
        return None


def _ndcg(actual_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    relevance_by_id = {
        candidate_id: len(expected_ids) - index for index, candidate_id in enumerate(expected_ids)
    }
    dcg = _dcg([relevance_by_id.get(candidate_id, 0) for candidate_id in actual_ids])
    ideal_dcg = _dcg(sorted(relevance_by_id.values(), reverse=True))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))
