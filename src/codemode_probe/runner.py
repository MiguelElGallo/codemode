from __future__ import annotations

import signal
import threading
from time import perf_counter

from codemode_probe.executors import CandidateExecutor
from codemode_probe.models import (
    ArmResult,
    ExecutionContext,
    ExecutionResult,
    FailureCategory,
    ProbeTask,
    TraceSummary,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.provenance import build_result_provenance
from codemode_probe.scoring import score_answer
from codemode_probe.workload import generate_candidates


class BenchmarkRunner:
    def __init__(self, executor: CandidateExecutor) -> None:
        self.executor = executor

    def run_task(
        self,
        task: ProbeTask,
        *,
        repetition: int = 1,
        context: ExecutionContext | None = None,
    ) -> ArmResult:
        started = perf_counter()
        execution, timed_out = _execute_with_timeout(self.executor, task, context)
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
            timed_out=timed_out,
        )

        return ArmResult(
            task_id=task.id,
            arm_name=self.executor.name,
            repetition=repetition,
            latency_ms=round(latency_ms, 3),
            timed_out=timed_out,
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


class _TaskTimeout(BaseException):
    pass


def _execute_with_timeout(
    executor: CandidateExecutor,
    task: ProbeTask,
    context: ExecutionContext | None = None,
) -> tuple[ExecutionResult, bool]:
    if not _can_use_signal_timeout():
        return _execute_candidate(executor, task, context), False

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    timeout_started = perf_counter()
    signal.signal(signal.SIGALRM, _raise_task_timeout)
    signal.setitimer(signal.ITIMER_REAL, task.timeout_seconds)
    try:
        return _execute_candidate(executor, task, context), False
    except _TaskTimeout:
        return (
            ExecutionResult(
                trace=TraceSummary(failure_category=FailureCategory.TIMEOUT),
                error="timeout",
            ),
            True,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed_seconds = perf_counter() - timeout_started
            remaining_seconds = max(1e-6, previous_timer[0] - elapsed_seconds)
            signal.setitimer(
                signal.ITIMER_REAL,
                remaining_seconds,
                previous_timer[1],
            )


def _can_use_signal_timeout() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


def _execute_candidate(
    executor: CandidateExecutor,
    task: ProbeTask,
    context: ExecutionContext | None,
) -> ExecutionResult:
    if context is None:
        return executor.execute(task)
    return executor.execute(task, context=context)


def _raise_task_timeout(signum, frame) -> None:
    raise _TaskTimeout("task execution timed out")
