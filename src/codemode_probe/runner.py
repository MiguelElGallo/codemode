from __future__ import annotations

from time import perf_counter

from codemode_probe.executors import CandidateExecutor
from codemode_probe.models import ArmResult, ProbeTask
from codemode_probe.oracle import rank_candidates
from codemode_probe.scoring import score_answer
from codemode_probe.workload import generate_candidates


class BenchmarkRunner:
    def __init__(self, executor: CandidateExecutor) -> None:
        self.executor = executor

    def run_task(self, task: ProbeTask, *, repetition: int = 1) -> ArmResult:
        started = perf_counter()
        execution = self.executor.execute(task)
        latency_ms = (perf_counter() - started) * 1000

        oracle = rank_candidates(
            task.id,
            generate_candidates(task.workload),
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
            execution=execution,
            score=score,
        )

    def run(self, tasks: list[ProbeTask], *, repetitions: int = 1) -> list[ArmResult]:
        results: list[ArmResult] = []
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                results.append(self.run_task(task, repetition=repetition))
        return results
