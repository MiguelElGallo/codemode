from __future__ import annotations

from time import perf_counter

from codemode_probe.executors import CandidateExecutor
from codemode_probe.models import ArmResult, ProbeTask
from codemode_probe.oracle import rank_candidates
from codemode_probe.provenance import build_result_provenance
from codemode_probe.scoring import score_answer
from codemode_probe.workload import generate_candidates


class BenchmarkRunner:
    def __init__(self, executor: CandidateExecutor) -> None:
        self.executor = executor

    def run_task(self, task: ProbeTask, *, repetition: int = 1) -> ArmResult:
        started = perf_counter()
        execution = self.executor.execute(task)
        latency_ms = (perf_counter() - started) * 1000

        candidates = generate_candidates(task.workload)
        oracle = rank_candidates(
            task.id,
            candidates,
            task.workload.top_k,
        )
        score = score_answer(
            execution.answer.model_dump() if execution.answer is not None else {},
            oracle,
        )

        return ArmResult(
            task_id=task.id,
            arm_name=self.executor.name,
            repetition=repetition,
            latency_ms=round(latency_ms, 3),
            provenance=build_result_provenance(
                task,
                executor_name=self.executor.name,
                executor_config=_executor_config(self.executor),
                candidates=candidates,
                oracle_answer=oracle,
            ),
            execution=execution,
            score=score,
        )

    def run(self, tasks: list[ProbeTask], *, repetitions: int = 1) -> list[ArmResult]:
        results: list[ArmResult] = []
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                results.append(self.run_task(task, repetition=repetition))
        return results


def _executor_config(executor: CandidateExecutor) -> dict[str, object]:
    config_metadata = getattr(executor, "config_metadata", None)
    if config_metadata is None:
        return {}
    config = config_metadata() if callable(config_metadata) else config_metadata
    if not isinstance(config, dict):
        return {"value": config}
    return config
