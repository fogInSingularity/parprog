#!/usr/bin/env python3
"""Aggregate raw benchmark CSV and plot time versus size for every configuration."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


PARALLEL = {"parallel", "combined", "combined-no-strassen"}
ALGORITHMS = {"naive", "transposed", "blocked", "strassen", "simd"} | PARALLEL


@dataclass(frozen=True, order=True)
class Key:
    algorithm: str
    n: int
    threads: int
    block: int
    cutoff: int
    seed: int
    values: str = "signed"  # Older CSV files used [-1,1] and had no range column.

    @classmethod
    def from_row(cls, row):
        return cls(
            algorithm=row["algorithm"],
            n=int(row["n"]),
            threads=int(row["threads"]),
            block=int(row["block"]),
            cutoff=int(row["cutoff"]),
            seed=int(row["seed"]),
            values=row.get("values", "signed"),
        )

    def serial(self):
        return Key(self.algorithm, self.n, 1, self.block, self.cutoff, self.seed, self.values)


def summarize(path):
    groups = {}
    repeats = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = Key.from_row(row)
            if key.values not in {"positive", "signed"}:
                raise ValueError(f"Unknown input range: {key.values}")
            seconds = float(row["seconds"])
            if key.algorithm not in ALGORITHMS:
                raise ValueError(f"Unknown algorithm: {key.algorithm}")
            if min(key.n, key.threads, key.block, key.cutoff) <= 0:
                raise ValueError(f"Nonpositive experiment parameter: {key}")
            if key.algorithm not in PARALLEL and key.threads != 1:
                raise ValueError(f"Serial algorithm with threads != 1: {key}")
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid time: {seconds}")
            error = float(row["max_scaled_error"])
            if not math.isfinite(error) or not 0 <= error <= 1:
                raise ValueError(f"Failed numerical validation: {key}")
            repeat = int(row["repeat"])
            if repeat < 0 or repeat in repeats.setdefault(key, set()):
                raise ValueError(f"Invalid or duplicate repeat: {key}, {repeat}")
            repeats[key].add(repeat)
            groups.setdefault(key, []).append(seconds)
    if not groups:
        raise ValueError("No measurements in CSV")
    results = {}
    for key, times in sorted(groups.items()):
        if repeats[key] != set(range(len(times))):
            raise ValueError(f"Missing repetition: {key}")
        results[key] = {
            "samples": len(times),
            "mean_seconds": statistics.mean(times),
            "stdev_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "speedup": None,
            "efficiency": None,
        }
    for key, result in results.items():
        if key.algorithm in PARALLEL:
            if key.serial() not in results:
                raise ValueError(f"Missing one-thread baseline for {key}")
            result["speedup"] = results[key.serial()]["mean_seconds"] / result["mean_seconds"]
            result["efficiency"] = result["speedup"] / key.threads
    return results


def write_summary(results, destination):
    with Path(destination).open("w", newline="", encoding="utf-8") as output:
        fields = [*Key.__dataclass_fields__, *next(iter(results.values()))]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            writer.writerow({**vars(key), **result})


def render(results, destination, title):
    # Import only for rendering: CSV analysis/tests need no third-party packages.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    series = {}
    for key, result in results.items():
        identity = (key.algorithm, key.threads, key.block, key.cutoff, key.seed, key.values)
        series.setdefault(identity, []).append((key.n, result))

    configs = {(key.block, key.cutoff, key.seed, key.values) for key in results}
    palette = plt.get_cmap("tab20").colors
    colors = {
        identity: palette[index % len(palette)]
        for index, identity in enumerate(sorted(series))
    }
    fig, axis = plt.subplots(figsize=(12, 7.5), layout="constrained")
    for (algorithm, threads, block, cutoff, seed, values), points in sorted(series.items()):
        points = sorted(points)
        description = f"{algorithm}, p={threads}"
        if len(configs) > 1:
            description += f", b={block}, cutoff={cutoff}, seed={seed}, values={values}"
        x = [n for n, _ in points]
        y = [result["mean_seconds"] for _, result in points]
        color = colors[(algorithm, threads, block, cutoff, seed, values)]
        axis.errorbar(
            x,
            y,
            yerr=[result["stdev_seconds"] for _, result in points],
            marker="o",
            capsize=3,
            color=color,
            label=description,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    sizes = sorted({key.n for key in results})
    axis.set_xticks(sizes, [str(n) for n in sizes])
    axis.set_xlabel("Размер матрицы N × N")
    axis.set_ylabel("Среднее время, с")
    if title:
        axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"time.{extension}", dpi=160)
    plt.close(fig)


def thread_counts(text):
    try:
        values = {int(item) for item in text.split(",")}
        if not values or min(values) <= 0:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Thread counts must be positive integers separated by commas"
        ) from error
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--title")
    parser.add_argument(
        "--threads",
        type=thread_counts,
        default={1, 2, 8, 32},
        help="parallel configurations to show (default: 1,2,8,32)",
    )
    arguments = parser.parse_args()
    results = summarize(arguments.csv)
    results = {
        key: result
        for key, result in results.items()
        if key.algorithm not in PARALLEL or key.threads in arguments.threads
    }
    if not results:
        parser.error("No measurements match the selected thread counts")
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_summary(results, arguments.output / "summary.csv")
    render(results, arguments.output, arguments.title)
    print(f"Saved summary and time graph to {arguments.output}")


if __name__ == "__main__":
    main()
