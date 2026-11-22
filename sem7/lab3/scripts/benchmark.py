#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def read_solution(path):
    with path.open(newline="") as stream:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(stream)]


def difference(left, right):
    if len(left) != len(right):
        raise ValueError("Different solution lengths")
    maximum = 0.0
    for (x, y), (other_x, other_y) in zip(left, right):
        if abs(x - other_x) > 1e-12 or not math.isfinite(y) or not math.isfinite(other_y):
            raise ValueError(f"Invalid solution pair at x={x}")
        maximum = max(maximum, abs(y - other_y))
    return maximum


def run(args, base, cpus, ranks, *options):
    launcher = [*base, "-n", str(ranks)]
    env = os.environ.copy()
    if cpus:
        mask = ",".join(map(str, cpus[:ranks]))
        env["PINNED_CPUS"] = mask
        launcher = ["taskset", "-c", mask, *launcher, str(Path(__file__).with_name("rank-on-cpu.sh"))]
    output = subprocess.check_output(
        [*launcher, str(args.binary.resolve()), *map(str, options)], env=env, text=True
    )
    print(output, end="", flush=True)
    if cpus:
        affinity = {tuple(map(int, match)) for match in re.findall(r"^MPI affinity: rank=(\d+) cpu=(\d+)$", output, re.M)}
        if affinity != set(enumerate(cpus[:ranks])):
            raise ValueError(f"Wrong rank binding: {affinity}")
    summary = next(line for line in output.splitlines() if line.startswith("solver="))
    samples = next(line for line in output.splitlines() if line.startswith("samples "))
    return dict(field.split("=", 1) for field in summary.split()), list(map(float, samples.split()[1:]))


def main():
    parser = argparse.ArgumentParser(description="Benchmark the V4 Numerov boundary problem.")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", default="4,8,16,24,28,32")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--pin", action="store_true")
    args = parser.parse_args()
    workers = sorted(set(map(int, args.workers.split(","))))
    if not workers or min(workers) < 2 or args.repeats < 1:
        parser.error("Pass parallel worker counts and at least one repeat")
    cpus = sorted(os.sched_getaffinity(0)) if args.pin else []
    if cpus and max(workers) > len(cpus):
        parser.error("More workers than allowed CPUs")

    version = subprocess.check_output([args.mpiexec, "--version"], text=True)
    openmpi = any(name in version for name in ("Open MPI", "OpenRTE", "PRRTE"))
    if args.pin and not openmpi:
        parser.error("--pin requires Open MPI")
    base = [args.mpiexec]
    if openmpi:
        base.extend(("--bind-to", "none", "--oversubscribe"))

    args.output.mkdir(parents=True, exist_ok=True)
    data_fields = ("a", "intervals", "solver", "ranks", "repeat", "seconds", "iterations", "residual")
    comparison_fields = ("a", "intervals", "half_full_delta", "refinement_delta", "mpi_delta")
    with (
        tempfile.TemporaryDirectory() as directory,
        (args.output / "timings.csv").open("w", newline="") as timings,
        (args.output / "verification.csv").open("w", newline="") as comparisons,
    ):
        writer = csv.DictWriter(timings, fieldnames=data_fields)
        checks = csv.writer(comparisons)
        writer.writeheader()
        checks.writerow(comparison_fields)
        temporary = Path(directory) / "solution.csv"

        for a in (100, 1000, 10000, 100000, 1000000):
            intervals = max(12800, math.ceil(100 * math.sqrt(a)))
            ranks = [p for p in workers if intervals // p >= 400 and math.ceil(intervals / p) <= 4000]
            if not ranks:
                raise ValueError(f"No worker count fits {intervals} unknowns for a={a}")
            common = ("--a", a, "--intervals", intervals)
            solution = args.output / f"solution-{a}.csv"

            def measure(p, solver, path):
                meta, samples = run(
                    args, base, cpus, p, *common, "--solver", solver, "--repeats", args.repeats,
                    "--warmups", 1, "--output", path,
                )
                if len(samples) != args.repeats or any(not math.isfinite(t) or t <= 0 for t in samples):
                    raise ValueError(f"Unexpected timing samples: {samples}")
                writer.writerows(
                    {**{key: meta[key] for key in data_fields if key in meta}, "repeat": i, "seconds": t}
                    for i, t in enumerate(samples)
                )
                timings.flush()
                return statistics.median(samples)

            reference = measure(1, "thomas", solution)
            expected = read_solution(solution)
            run(args, base, cpus, 1, "--solver", "thomas", "--a", a, "--full",
                "--intervals", 2 * intervals, "--output", temporary)
            half_full = difference(expected, read_solution(temporary))
            run(args, base, cpus, 1, "--solver", "thomas", "--a", a,
                "--intervals", 2 * intervals, "--output", temporary)
            refinement = difference(expected, read_solution(temporary)[::2])
            mpi_delta = 0.0

            for p in ranks:
                elapsed = measure(p, "reduction", temporary)
                delta = difference(expected, read_solution(temporary))
                mpi_delta = max(mpi_delta, delta)
                print(f"a={a} ranks={p} speedup={reference / elapsed:.3f} delta={delta:.3g}", flush=True)
            if max(half_full, mpi_delta) > 1e-10:
                raise ValueError(f"Parallel or mirrored solution differs for a={a}")
            checks.writerow((a, intervals, half_full, refinement, mpi_delta))
            comparisons.flush()


if __name__ == "__main__":
    main()
