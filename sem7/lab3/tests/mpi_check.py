import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def solve(launcher, flags, binary, ranks, path, *args):
    subprocess.run(
        [launcher, *flags, "-n", str(ranks), binary, *args, "--output", str(path)],
        check=True,
    )
    with path.open(newline="") as stream:
        return [(float(row["x"]), float(row["y"])) for row in csv.DictReader(stream)]


def check_same(left, right):
    if len(left) != len(right):
        raise ValueError(f"Different grid lengths: {len(left)}, {len(right)}")
    for (x_left, y_left), (x_right, y_right) in zip(left, right):
        if not math.isclose(x_left, x_right, abs_tol=1e-13) or not math.isclose(
            y_left, y_right, rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"Different nodes: {(x_left, y_left)}, {(x_right, y_right)}")


def main():
    launcher, binary = sys.argv[1:]
    os.environ["PATH"] = str(Path(launcher).parent) + os.pathsep + os.environ["PATH"]
    version = subprocess.check_output([launcher, "--version"], text=True)
    flags = ["--bind-to", "none", "--oversubscribe"] if "Open MPI" in version else []

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "solution.csv"
        full = solve(
            launcher, flags, binary, 1, output, "--solver", "thomas", "--full", "--intervals", "2102"
        )
        half = solve(launcher, flags, binary, 4, output, "--intervals", "1051")
        check_same(full, half)

        # Blocks of one and two unknowns exercise both interface row counts.
        small = solve(launcher, flags, binary, 1, output, "--solver", "thomas", "--a", "0.001", "--intervals", "6")
        distributed = solve(launcher, flags, binary, 4, output, "--a", "0.001", "--intervals", "6")
        check_same(small, distributed)


if __name__ == "__main__":
    main()
