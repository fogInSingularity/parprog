#!/usr/bin/env python3
import subprocess
import sys
import matplotlib.pyplot as plt

BINARY = "./build/task3.x"
SIZES = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    30_000_000,
]
RUNS = 3


def measure(size: int) -> tuple[float, float]:
    pms_times, qs_times = [], []
    for _ in range(RUNS):
        result = subprocess.run(
            [BINARY, str(size)],
            capture_output=True, text=True, check=True,
        )
        t_pms, t_qs = map(float, result.stdout.split())
        pms_times.append(t_pms)
        qs_times.append(t_qs)
    return sum(pms_times) / RUNS, sum(qs_times) / RUNS


def main():
    pms_times, qs_times = [], []

    for size in SIZES:
        t_pms, t_qs = measure(size)
        pms_times.append(t_pms)
        qs_times.append(t_qs)
        print(f"n={size:>10,}  pms={t_pms:.4f}s  qsort={
              t_qs:.4f}s", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(SIZES, pms_times, marker="o",
            linewidth=2, label="parallel mergesort")
    ax.plot(SIZES, qs_times,  marker="s", linewidth=2, label="libc qsort")

    ax.set_xscale("log")
    ax.set_xlabel("Array size (elements)", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title(f"parallel mergesort vs libc qsort  (avg of {
                 RUNS} runs, -O2)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    for size, tp, tq in zip(SIZES, pms_times, qs_times):
        if tp > 0:
            ax.annotate(
                f"{tq/tp:.1f}×",
                xy=(size, tp),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=8, color="tab:blue",
            )

    fig.tight_layout()
    out = "sort_benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}", file=sys.stderr)
    plt.show()


if __name__ == "__main__":
    main()
