#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, output, name):
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=160)
    plt.close(fig)


def read_results(source):
    groups = {}
    with (source / "timings.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["a"]), int(row["intervals"]), row["solver"], int(row["ranks"]))
            groups.setdefault(key, []).append(float(row["seconds"]))
    baselines = {(a, n): statistics.median(times)
                 for (a, n, solver, _), times in groups.items() if solver == "thomas"}
    results = []
    for (a, n, solver, ranks), times in sorted(groups.items()):
        median = statistics.median(times)
        speedup = baselines[a, n] / median
        results.append({
            "a": a,
            "intervals": n,
            "solver": solver,
            "ranks": ranks,
            "samples": len(times),
            "mean_seconds": statistics.mean(times),
            "median_seconds": median,
            "stdev_seconds": statistics.stdev(times) if len(times) > 1 else 0,
            "speedup": speedup,
            "efficiency": speedup / ranks,
        })
    return results


def scaling(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for a in sorted({row["a"] for row in results}):
        points = sorted((row for row in results if row["a"] == a), key=lambda row: row["ranks"])
        ranks = [row["ranks"] for row in points]
        label = rf"$a=10^{{{len(str(a)) - 1}}}$"
        axes[0].plot(ranks, [1000 * row["median_seconds"] for row in points], "o-", label=label)
        axes[1].plot(ranks, [row["speedup"] for row in points], "o-")
        axes[2].plot(ranks, [row["efficiency"] for row in points], "o-")
    axes[0].set_ylabel("Медианное время, мс")
    axes[0].set_yscale("log")
    axes[0].legend(ncols=2)
    axes[1].set_ylabel("Ускорение")
    axes[2].set_ylabel("Эффективность")
    for axis in axes:
        axis.set_xticks((1, 4, 8, 16, 24, 32))
        axis.set_xlabel("Число MPI-рангов")
        axis.grid(alpha=0.25)
    save(fig, output, "scaling")


def solutions(source, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for a in (100, 1000, 10000, 100000, 1000000):
        with (source / f"solution-{a}.csv").open(newline="") as stream:
            x, y = zip(*((float(row["x"]), float(row["y"])) for row in csv.DictReader(stream)))
        label = rf"$a=10^{{{len(str(a)) - 1}}}$"
        axes[0].plot(x, y, label=label)
        offset, near = zip(*((10 - value, y[i]) for i, value in enumerate(x) if value >= 9))
        axes[1].plot(offset, near)
    axes[0].set_xlabel(r"$x$")
    axes[0].legend(ncols=2, loc="upper center")
    axes[1].set_xlabel("Расстояние до правой границы")
    axes[1].set_xscale("symlog", linthresh=1e-4)
    for axis in axes:
        axis.set_ylabel(r"$y(x)$")
        axis.grid(alpha=0.25)
    save(fig, output, "solution")


def main():
    parser = argparse.ArgumentParser(description="Plot the Numerov solution and MPI timings.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = read_results(args.source)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    scaling(results, args.output)
    solutions(args.source, args.output)


if __name__ == "__main__":
    main()
