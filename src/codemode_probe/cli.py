from __future__ import annotations

import argparse
from pathlib import Path

from codemode_probe.artifacts import create_run_dir, write_run_artifacts
from codemode_probe.executors import DeterministicOracleExecutor
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.workload import make_probe_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Code Mode probe slices.")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/outputs"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    task = make_probe_task("synthetic_fanout_smoke")
    runner = BenchmarkRunner(DeterministicOracleExecutor())
    results = runner.run([task], repetitions=args.repetitions)
    run_dir = create_run_dir(args.out, run_id=args.run_id)
    write_run_artifacts(run_dir, [task], results)
    print(run_dir)


if __name__ == "__main__":
    main()
