import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot.py"
SPEC = importlib.util.spec_from_file_location("loop_plot", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


class PlotTests(unittest.TestCase):
    def analyze(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(
                    ["task", "backend", "rows", "cols", "workers", "repeat", "seconds", "checksum"]
                )
                writer.writerows(rows)
            return PLOT.summarize(path)

    def test_best_sequential_baseline(self):
        rows = [
            ["3v", "seq", 7, 11, 1, 0, 10, 0],
            ["3v", "fused", 7, 11, 1, 0, 5, 0],
            ["3v", "threads", 7, 11, 2, 0, 2.5, 0],
            ["3v", "mpi", 7, 11, 2, 0, 2, 0],
        ]
        results = self.analyze(rows)
        threaded = results[PLOT.Key("3v", "threads", 7, 11, 2)]
        self.assertEqual(threaded["baseline"], "fused")
        self.assertEqual(threaded["speedup"], 2)
        self.assertEqual(threaded["efficiency"], 1)
        self.assertEqual(results[PLOT.Key("3v", "mpi", 7, 11, 2)]["speedup"], 2.5)

    def test_shapes_and_tasks_are_separate(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.analyze([
                ["reference", "seq", 7, 11, 1, 0, 5, 0],
                ["3v", "mpi", 7, 11, 2, 0, 2, 0],
            ])
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.analyze([
                ["3v", "seq", 11, 7, 1, 0, 5, 0],
                ["3v", "mpi", 7, 11, 2, 0, 2, 0],
            ])

    def test_invalid_measurements(self):
        for seconds in (0, -1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.analyze([["3v", "seq", 7, 11, 1, 0, seconds, 0]])
        row = ["3v", "seq", 7, 11, 1, 0, 1, 0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.analyze([row, row])
        row[5] = 1
        with self.assertRaisesRegex(ValueError, "Missing repetition"):
            self.analyze([row])


if __name__ == "__main__":
    unittest.main()
