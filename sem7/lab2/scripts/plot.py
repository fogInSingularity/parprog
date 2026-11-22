#!/usr/bin/env python3
"""Plot timings, speedup, and efficiency for the loop lab."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


TASK_LABELS = {
    "reference": "Эталон",
    "1v": "Вариант 1в",
    "2g": "Вариант 2г",
    "3d": "Вариант 3д",
    "3v": "Вариант 3в",  # Archived measurements
}
BACKEND_STYLES = {"threads": ("Потоки", "-"), "mpi": ("MPI", "--"), "openmp": ("OpenMP", "-.")}
SEQUENTIAL_BACKENDS = {"seq", "fused"}
SCALING_METRICS = (
    ("speedup", "Ускорение S(p) = Tseq / Tp"),
    ("efficiency", "Эффективность E(p) = S(p) / p"),
)


@dataclass(frozen=True, order=True)
class Key:
    task: str
    backend: str
    rows: int
    cols: int
    workers: int

    @classmethod
    def from_row(cls, row):
        return cls(
            task=row["task"],
            backend=row["backend"],
            rows=int(row["rows"]),
            cols=int(row["cols"]),
            workers=int(row["workers"]),
        )

    def problem(self):
        return self.task, self.rows, self.cols


def summarize(path):
    groups = {}
    repeats = {}
    with Path(path).open(newline="") as source:
        for row in csv.DictReader(source):
            key = Key.from_row(row)
            if key.task not in TASK_LABELS or (
                key.backend not in SEQUENTIAL_BACKENDS and key.backend not in BACKEND_STYLES
            ):
                raise ValueError(f"Unknown configuration: {key}")
            if min(key.rows, key.cols, key.workers) <= 0:
                raise ValueError(f"Nonpositive configuration: {key}")
            if key.backend in SEQUENTIAL_BACKENDS and key.workers != 1:
                raise ValueError(f"Sequential backend with multiple workers: {key}")
            seconds = float(row["seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid measurement: {seconds}")
            repeat = int(row["repeat"])
            if repeat < 0 or repeat in repeats.setdefault(key, set()):
                raise ValueError(f"Invalid or duplicate repeat: {key}")
            repeats[key].add(repeat)
            groups.setdefault(key, []).append(seconds)
    if not groups:
        raise ValueError("No measurements in CSV")
    results = {}
    for key, seconds in sorted(groups.items()):
        if repeats[key] != set(range(len(seconds))):
            raise ValueError(f"Missing repetition: {key}")
        results[key] = {
            "samples": len(seconds),
            "mean_seconds": statistics.mean(seconds),
            "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0,
            "baseline": "",
            "baseline_seconds": None,
            "speedup": None,
            "efficiency": None,
        }
    add_speedups(results)
    return results


def add_speedups(results):
    baselines = {}
    for key, result in results.items():
        if key.backend in SEQUENTIAL_BACKENDS:
            fastest = baselines.get(key.problem())
            seconds = result["mean_seconds"]
            if fastest is None or seconds < fastest[1]:
                baselines[key.problem()] = (key.backend, seconds)

    for key, result in results.items():
        if key.backend not in BACKEND_STYLES:
            continue
        baseline = baselines.get(key.problem())
        if baseline is None:
            raise ValueError(f"Missing sequential baseline: {key}")
        result["baseline"], result["baseline_seconds"] = baseline
        result["speedup"] = result["baseline_seconds"] / result["mean_seconds"]
        result["efficiency"] = result["speedup"] / key.workers


def write_summary(results, path):
    with Path(path).open("w", newline="") as output:
        fields = [*Key.__dataclass_fields__, *next(iter(results.values()))]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            writer.writerow({**vars(key), **result})


def save_figure(fig, directory, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(Path(directory) / f"{stem}.{suffix}", dpi=160, bbox_inches="tight")


def colors_by_key(plt, keys):
    palette = plt.get_cmap("tab10").colors
    return {key: palette[index % len(palette)] for index, key in enumerate(keys)}


def worker_ticks(counts):
    if len(counts) <= 8:
        return counts
    step = math.ceil(len(counts) / 8)
    return [counts[0], *counts[step - 1 : -1 : step], counts[-1]]


def draw_scaling(plt, series, tasks, shapes, counts):
    colors = colors_by_key(plt, shapes)
    legend_height = 0.3 * len(shapes) + 0.2
    fig = plt.figure(figsize=(13, 4 * len(tasks) + legend_height), layout="constrained")
    grid = fig.add_gridspec(
        len(tasks) + 1,
        2,
        height_ratios=[4] * len(tasks) + [legend_height],
    )
    task_axes = {
        task: [fig.add_subplot(grid[row, column]) for column in range(2)]
        for row, task in enumerate(tasks)
    }
    legend_axis = fig.add_subplot(grid[-1, :])
    legend_axis.set_axis_off()

    legend = {}
    for (task, backend, rows, cols), points in sorted(series.items()):
        backend_label, line_style = BACKEND_STYLES[backend]
        label = f"{backend_label}, {rows}×{cols}"
        workers, measurements = zip(*sorted(points))
        for axis, (metric, _) in zip(task_axes[task], SCALING_METRICS):
            (line,) = axis.plot(
                workers,
                [value[metric] for value in measurements],
                marker="o",
                markersize=3 if len(counts) > 8 else 6,
                label=label,
                color=colors[rows, cols],
                linestyle=line_style,
            )
            legend.setdefault(label, line)

    for task, axes in task_axes.items():
        for axis in axes:
            if len(counts) <= 8:
                axis.set_xscale("log", base=2)
            axis.set_xticks(worker_ticks(counts))
            axis.set_xlabel("Число исполнителей")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(f"{TASK_LABELS[task]}\n{SCALING_METRICS[0][1]}")
        axes[1].set_ylabel(SCALING_METRICS[1][1])
        axes[1].axhline(1, color="gray", linestyle=":", linewidth=1)
    legend_axis.legend(legend.values(), legend.keys(), loc="center", ncol=2)
    return fig


def size_on_axis(rows, cols, square):
    return rows if square else rows * cols


def time_points(results, task, square):
    parallel = {}
    sequential = {}
    for key, value in results.items():
        if key.task != task:
            continue
        size = size_on_axis(key.rows, key.cols, square)
        milliseconds = value["mean_seconds"] * 1000
        if key.backend in BACKEND_STYLES:
            parallel.setdefault((key.backend, key.workers), []).append((size, milliseconds))
        else:
            sequential[size] = min(sequential.get(size, math.inf), milliseconds)
    return parallel, sequential


def draw_times(plt, results, tasks, shapes, counts):
    colors = colors_by_key(plt, counts)
    square = all(rows == cols for rows, cols in shapes)
    size_label = "Сторона квадратного массива N" if square else "Число элементов массива"
    ticks = sorted({size_on_axis(rows, cols, square) for rows, cols in shapes})
    fig = plt.figure(figsize=(max(13, 4.5 * len(tasks)), 6.5), layout="constrained")
    grid = fig.add_gridspec(2, len(tasks), height_ratios=(4.5, 2))
    axes = [fig.add_subplot(grid[0, column]) for column in range(len(tasks))]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.set_axis_off()

    legend = {}
    for task, axis in zip(tasks, axes):
        parallel, baseline = time_points(results, task, square)
        for backend, (backend_label, line_style) in BACKEND_STYLES.items():
            for count in counts:
                points = parallel.get((backend, count))
                if not points:
                    continue
                label = f"{backend_label}, p={count}"
                (line,) = axis.plot(
                    *zip(*sorted(points)),
                    marker="o",
                    color=colors[count],
                    linestyle=line_style,
                )
                legend.setdefault(label, line)

        (line,) = axis.plot(
            *zip(*sorted(baseline.items())), color="black", marker="s", linewidth=2
        )
        legend.setdefault("Последовательная (лучшая)", line)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks, [str(value) for value in ticks])
        axis.set_yscale("log")
        axis.set_xlabel(f"{TASK_LABELS[task]} · {size_label}")
        axis.set_ylabel("Время, мс (логарифмическая шкала)")
        axis.grid(alpha=0.25, which="both")
    legend_axis.legend(legend.values(), legend.keys(), loc="center", ncol=3)
    return fig


def draw_time_heatmaps(plt, results, tasks, shapes, counts):
    from matplotlib.colors import LogNorm

    backends_by_task = {task: set() for task in tasks}
    for key in results:
        if key.backend in BACKEND_STYLES:
            backends_by_task[key.task].add(key.backend)
    columns = max(map(len, backends_by_task.values()))
    fig = plt.figure(figsize=(13, 2.55 * len(tasks) + 1.4), layout="constrained")
    panels = fig.add_gridspec(len(tasks), columns + 1, width_ratios=[1] * columns + [0.035])
    times = [value["mean_seconds"] * 1000 for value in results.values()]
    norm = LogNorm(vmin=min(times), vmax=max(times))
    square = all(rows == cols for rows, cols in shapes)
    sizes = [size_on_axis(rows, cols, square) for rows, cols in shapes]

    for row, task in enumerate(tasks):
        parallel, baseline = time_points(results, task, square)
        series = {key: dict(points) for key, points in parallel.items()}
        backends = [backend for backend in BACKEND_STYLES if backend in backends_by_task[task]]
        for column, backend in enumerate(backends):
            panel = panels[row, :-1] if len(backends) == 1 else panels[row, column]
            axis = fig.add_subplot(panel)
            timings = [baseline, *(series.get((backend, count), {}) for count in counts)]
            grid = [[values.get(size, math.nan) for values in timings] for size in sizes]
            image = axis.imshow(
                grid,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis_r",
                norm=norm,
            )
            axis.axvline(0.5, color="white", linewidth=1)
            axis.set_yticks(range(len(shapes)), [f"{rows}×{cols}" for rows, cols in shapes])
            axis.set_xticks(range(len(counts) + 1), [r"$T_1$", *map(str, counts)], fontsize=8)
            axis.set_xlabel(f"{BACKEND_STYLES[backend][0]} · число исполнителей")
            if column == 0:
                axis.set_ylabel(TASK_LABELS[task])
            else:
                axis.tick_params(labelleft=False)

    fig.colorbar(image, cax=fig.add_subplot(panels[:, -1]), label="Время, мс")
    return fig


def render(results, directory, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for key, result in results.items():
        if key.backend in BACKEND_STYLES:
            identity = (key.task, key.backend, key.rows, key.cols)
            series.setdefault(identity, []).append((key.workers, result))
    if not series:
        raise ValueError("No parallel measurements to plot")

    measured_tasks = {task for task, _, _, _ in series}
    tasks = [task for task in TASK_LABELS if task in measured_tasks]
    shapes = sorted({(rows, cols) for _, _, rows, cols in series})
    counts = sorted({key.workers for key in results if key.backend in BACKEND_STYLES})

    scaling = draw_scaling(plt, series, tasks, shapes, counts)
    if title:
        scaling.suptitle(title)
    save_figure(scaling, directory, "scaling")
    plt.close(scaling)

    if len(counts) > 10:
        times = draw_time_heatmaps(plt, results, tasks, shapes, counts)
        times_title = "время по размерам и числу исполнителей"
    else:
        times = draw_times(plt, results, tasks, shapes, counts)
        times_title = "время от размера"
    if title:
        times.suptitle(f"{title}: {times_title}")
    save_figure(times, directory, "time")
    plt.close(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--title")
    args = parser.parse_args()
    results = summarize(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)
    write_summary(results, args.output / "summary.csv")
    render(results, args.output, args.title)
    print(f"Saved scaling/time PNG and PDF, summary.csv in {args.output}")


if __name__ == "__main__":
    main()
