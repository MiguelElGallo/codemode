from __future__ import annotations

from codemode_probe.calibration import run_scoring_calibration_checks


def test_run_scoring_calibration_checks_covers_known_negative_and_degraded_cases() -> None:
    results = run_scoring_calibration_checks()

    assert [result.name for result in results] == [
        "calibration_schema_invalid",
        "calibration_task_id_mismatch",
        "calibration_partial_overlap",
        "calibration_hallucinated_ids",
        "calibration_duplicate_ids",
        "calibration_degraded_ranking",
    ]
    assert all(result.passed for result in results)

    by_name = {result.name: result for result in results}
    assert by_name["calibration_schema_invalid"].details["failure_reason"] == "schema_invalid"
    assert by_name["calibration_task_id_mismatch"].details["failure_reason"] == "task_id_mismatch"
    assert by_name["calibration_hallucinated_ids"].details["top_k_overlap"] == 0.0
    assert by_name["calibration_duplicate_ids"].details["top_k_overlap"] == 2 / 3
    assert (
        by_name["calibration_degraded_ranking"].details["degraded"]["ndcg_at_k"]
        < by_name["calibration_degraded_ranking"].details["perfect"]["ndcg_at_k"]
    )
