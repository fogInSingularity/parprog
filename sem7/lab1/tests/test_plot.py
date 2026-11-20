import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot.py"
SPEC = importlib.util.spec_from_file_location("matrix_plot", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


class AnalysisTests(unittest.TestCase):
    def analyze(self, rows, with_values=False):
        fields = [
            "algorithm", "n", "threads", "block", "cutoff", "seed", "repeat",
            "seconds", "max_scaled_error",
        ]
        if with_values:
            fields.append("values")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(fields)
                writer.writerows(rows)
            return PLOT.summarize(path)

    def test_speedup_uses_same_algorithm_at_one_thread(self):
        rows = [
            ["naive", 8, 1, 4, 4, 42, 0, 100, 0],
            ["parallel", 8, 1, 4, 4, 42, 0, 8, 0],
            ["parallel", 8, 1, 4, 4, 42, 1, 12, 0],
            ["parallel", 8, 2, 4, 4, 42, 0, 5, 0],
        ]
        results = self.analyze(rows)
        result = results[PLOT.Key("parallel", 8, 2, 4, 4, 42)]
        self.assertEqual(result["speedup"], 2)
        self.assertEqual(result["efficiency"], 1)

    def test_reject_missing_baseline(self):
        with self.assertRaisesRegex(ValueError, "one-thread baseline"):
            self.analyze([["combined", 8, 2, 4, 4, 42, 0, 5, 0]])

    def test_different_input_ranges_are_not_mixed(self):
        rows = [
            ["parallel", 8, 1, 4, 4, 42, 0, 10, 0, "signed"],
            ["parallel", 8, 2, 4, 4, 42, 0, 5, 0, "signed"],
            ["parallel", 8, 1, 4, 4, 42, 0, 12, 0, "positive"],
            ["parallel", 8, 2, 4, 4, 42, 0, 3, 0, "positive"],
        ]
        results = self.analyze(rows, with_values=True)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[PLOT.Key("parallel", 8, 2, 4, 4, 42, "signed")]["speedup"], 2)
        self.assertEqual(results[PLOT.Key("parallel", 8, 2, 4, 4, 42, "positive")]["speedup"], 4)

    def test_reject_invalid_measurements(self):
        for seconds, error in [(0, 0), (-1, 0), (float("nan"), 0), (1, 2), (1, float("nan"))]:
            with self.subTest(seconds=seconds, error=error), self.assertRaises(ValueError):
                self.analyze([["naive", 8, 1, 4, 4, 42, 0, seconds, error]])

    def test_reject_mixed_repetitions(self):
        row = ["naive", 8, 1, 4, 4, 42, 0, 1, 0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.analyze([row, row])
        row[6] = 1
        with self.assertRaisesRegex(ValueError, "Missing repetition"):
            self.analyze([row])


if __name__ == "__main__":
    unittest.main()
