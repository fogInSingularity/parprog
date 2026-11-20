#!/usr/bin/env python3
"""Run the reference and task 3v with shared memory and MPI, retaining every CSV."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import subprocess

import mpi


def positive_list(value):
    try:
        result = sorted({int(item) for item in value.split(",")})
        if not result or min(result) <= 0:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected positive integers separated by commas"
        ) from error
    return result


def merge_measurements(files, destination):
    with destination.open("w", newline="") as output:
        writer = None
        fields = None
        for path in files:
            if not path.is_file():
                continue
            with path.open(newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    fields = reader.fieldnames
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                if reader.fieldnames != fields:
                    raise ValueError(f"Incompatible CSV fields: {path}")
                writer.writerows(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    build_dir = Path(__file__).resolve().parents[1] / "build"
    parser.add_argument("--bin-dir", type=Path, default=build_dir)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--sizes", type=positive_list, default=[1000, 5000])
    parser.add_argument("--workers", type=positive_list, default=[1, 2, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mpiexec")
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("Repeats must be positive; warmups must be nonnegative")
    if 1 not in args.workers:
        args.workers.insert(0, 1)
    binary = (args.bin_dir / "loop_lab").resolve()
    parallel = (args.bin_dir / "loop_mpi").resolve()
    launcher = mpi.launcher_path(args.bin_dir, args.mpiexec)
    mpi_environment = mpi.environment(launcher)
    worker_list = ",".join(map(str, args.workers))
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = args.output / "measurements"
    measurements.mkdir(exist_ok=True)
    commands = []
    files = []
    failed = False

    def run(command, path, environment=None):
        nonlocal failed
        commands.append(command)
        files.append(path)
        print("$ " + shlex.join(command), flush=True)
        # A launcher can fail before the executable opens its CSV.
        path.unlink(missing_ok=True)
        # A failed configuration must not prevent later MPI counts from running.
        status = subprocess.run(command, env=environment).returncode
        failed |= status != 0

    for size in args.sizes:
        common = [
            "bench",
            "--rows", str(size),
            "--cols", str(size),
            "--repeats", str(args.repeats),
            "--warmups", str(args.warmups),
        ]
        path = measurements / f"local-{size}.csv"
        run([str(binary), *common, "--workers", worker_list, "--output", str(path)], path)
        for workers in args.workers:
            path = measurements / f"mpi-{size}-p{workers}.csv"
            invocation = mpi.command(launcher, workers, parallel, [*common, "--output", str(path)])
            run(invocation, path, mpi_environment)

    merge_measurements(files, args.output / "raw.csv")
    metadata = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "commands": commands,
        "mpi_launcher": str(launcher),
        "sizes": args.sizes,
        "workers": args.workers,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "failed": failed,
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output / 'raw.csv'}; failures={failed}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
