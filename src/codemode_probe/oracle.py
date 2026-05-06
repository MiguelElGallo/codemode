from __future__ import annotations

from codemode_probe.models import Candidate, RankedCandidate, StructuredAnswer


def candidate_score(candidate: Candidate) -> tuple[float, dict[str, float]]:
    if candidate.is_draft or candidate.is_bot_authored:
        return 0.0, {"excluded": 1.0}

    approval_score = min(candidate.approvals, 4) / 4
    ci_score = 1.0 if candidate.failing_checks == 0 else max(0.0, 1 - candidate.failing_checks / 4)
    reaction_score = min(candidate.reactions, 50) / 50
    recency_score = max(0.0, 1 - candidate.age_days / 60)
    size_score = max(0.0, 1 - candidate.changed_files / 100)

    breakdown = {
        "relevance": candidate.relevance * 0.35,
        "approvals": approval_score * 0.20,
        "ci": ci_score * 0.20,
        "reactions": reaction_score * 0.10,
        "recency": recency_score * 0.10,
        "size": size_score * 0.05,
    }
    return round(sum(breakdown.values()), 6), breakdown


def rank_candidates(task_id: str, candidates: list[Candidate], top_k: int) -> StructuredAnswer:
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        score, breakdown = candidate_score(candidate)
        if score <= 0:
            continue
        ranked.append(
            RankedCandidate(
                id=candidate.id,
                score=score,
                rationale=_rationale(candidate),
                score_breakdown={key: round(value, 6) for key, value in breakdown.items()},
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.id))
    return StructuredAnswer(task_id=task_id, candidates=ranked[:top_k])


def _rationale(candidate: Candidate) -> str:
    return (
        f"{candidate.approvals} approvals, {candidate.failing_checks} failing checks, "
        f"{candidate.reactions} reactions, {candidate.changed_files} changed files"
    )
