import argparse
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess


def launcher_path(bin_dir, explicit=None):
    candidate = explicit
    config = Path(bin_dir) / "mpi-launcher.txt"
    if not candidate and config.is_file():
        candidate = config.read_text().strip()
    if not candidate or candidate.endswith("-NOTFOUND"):
        candidate = shutil.which("mpirun") or shutil.which("mpiexec")
    if not candidate:
        raise ValueError("MPI launcher not found; configure MPIEXEC_EXECUTABLE or pass --mpiexec")
    path = Path(candidate).absolute()
    if not path.is_file():
        raise ValueError(f"MPI launcher does not exist: {path}")
    return path


def environment(launcher):
    result = os.environ.copy()
    directory = str(Path(launcher).parent)
    result["PATH"] = directory + os.pathsep + result.get("PATH", "")
    prterun = Path(directory) / "prterun"
    if prterun.is_file():
        result.setdefault("OMPI_PRTERUN", str(prterun))
    return result


@lru_cache(maxsize=None)
def launcher_flags(launcher):
    version = subprocess.run(
        [str(launcher), "--version"],
        env=environment(launcher),
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    description = version.stdout + version.stderr
    if any(marker in description for marker in ("Open MPI", "OpenRTE", "PRRTE")):
        return ["--bind-to", "none", "--oversubscribe"]
    return []


def command(launcher, workers, binary, arguments):
    return [str(launcher), *launcher_flags(launcher), "-np", str(workers), str(binary), *arguments]


def main():
    parser = argparse.ArgumentParser(
        description="Resolve an MPI launcher and its environment without changing the login shell."
    )
    parser.add_argument("-n", "--workers", type=int, default=2)
    parser.add_argument("--mpiexec")
    parser.add_argument("binary", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("Worker count must be positive")
    binary = args.binary.resolve()
    launcher = launcher_path(binary.parent, args.mpiexec)
    invocation = command(launcher, args.workers, binary, args.arguments)
    return subprocess.run(invocation, env=environment(launcher)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
