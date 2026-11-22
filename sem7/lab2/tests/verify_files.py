#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mpi


def main():
    parser = argparse.ArgumentParser(
        description="Compare literal sequential output with fused, C++ thread, and MPI files."
    )
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
        outputs = {
            "reference": ["fused-p1", "threads-p1", "threads-p2", "threads-p8"],
            "1v": ["fused-p1"],
            "2g": ["fused-p1", "openmp-p1", "openmp-p2", "openmp-p8"],
            "3d": ["fused-p1"],
        }
        if args.ranks is not None:
            launcher = mpi.launcher_path(directory)
            invocation = mpi.command(launcher, args.ranks, directory / "loop_mpi", common)
            subprocess.run(invocation, env=mpi.environment(launcher), check=True)
            for task in ("reference", "1v", "3d"):
                outputs[task].append(f"mpi-p{args.ranks}")
        for task, backends in outputs.items():
            prefix = f"{task}-{args.rows}x{args.cols}-"
            reference = (output / f"{prefix}seq-p1.txt").read_bytes()
            height, width = (args.cols, args.rows) if task == "1v" else (args.rows, args.cols)
            lines = reference.splitlines()
            if len(lines) != height or any(len(line.split()) != width for line in lines):
                raise ValueError(f"Unexpected result shape for {task}")
            for backend in backends:
                path = output / f"{prefix}{backend}.txt"
                if path.read_bytes() != reference:
                    raise ValueError(f"File differs from sequential: {path.name}")
        print("All full-precision output files are byte-for-byte identical.")


if __name__ == "__main__":
    main()
