#!/usr/bin/env python3
"""Compare literal sequential output with fused, C++ thread, and MPI files."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mpi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=11)
    args = parser.parse_args()
    if args.ranks is not None and args.ranks <= 0:
        parser.error("Rank count must be positive")
    directory = args.bin_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="loop-files-") as temporary:
        output = Path(temporary)
        common = [
            "check",
            "--rows", str(args.rows),
            "--cols", str(args.cols),
            "--dump", str(output),
        ]
        subprocess.run([str(directory / "loop_lab"), *common, "--workers", "1,2,8"], check=True)
        outputs = ["fused-p1", "threads-p1", "threads-p2", "threads-p8"]
        if args.ranks is not None:
            launcher = mpi.launcher_path(directory)
            invocation = mpi.command(launcher, args.ranks, directory / "loop_mpi", common)
            subprocess.run(invocation, env=mpi.environment(launcher), check=True)
            outputs.append(f"mpi-p{args.ranks}")
        for task in ("reference", "3v"):
            prefix = f"{task}-{args.rows}x{args.cols}-"
            reference = (output / f"{prefix}seq-p1.txt").read_bytes()
            for backend in outputs:
                path = output / f"{prefix}{backend}.txt"
                if path.read_bytes() != reference:
                    raise ValueError(f"File differs from sequential: {path.name}")
        print("All full-precision output files are byte-for-byte identical.")


if __name__ == "__main__":
    main()
