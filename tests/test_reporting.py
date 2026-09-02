import csv
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from lgts_benchmark.reporting import _live_summary, build_aggregate_reports, initialize_live_workbook


class ValidationFirstReportingTests(unittest.TestCase):
    def test_dashboard_ranking_uses_validation_not_test_f1(self):
        common = {
            "dataset": "demo",
            "model": "ResNet18",
            "activation_scope": "full_backbone",
            "mean_epoch_time_sec": 1.0,
            "test_error_rate_percent": 10.0,
        }
        rows = [
            {**common, "activation_variant": "validation_winner", "best_validation_f1": 0.90, "test_f1_macro": 0.70},
            {**common, "activation_variant": "test_winner", "best_validation_f1": 0.80, "test_f1_macro": 0.95},
        ]
        summaries = _live_summary(rows)
        self.assertEqual(summaries[0]["activation_variant"], "validation_winner")

    def test_sealed_selection_rows_do_not_require_test_metrics(self):
        rows = [{
            "dataset": "demo", "model": "ResNet18", "activation_scope": "full_backbone",
            "activation_variant": "LGTS-f2-Slope", "best_validation_f1": 0.91,
            "validation_f1_macro": 0.91, "validation_accuracy": 0.92,
            "mean_epoch_time_sec": 1.0,
        }]
        summary = _live_summary(rows)[0]
        self.assertEqual(summary["validation_f1_mean"], 0.91)
        self.assertEqual(summary["test_f1_mean"], "")

    def test_aggregate_reports_use_validation_when_test_is_sealed(self):
        rows = []
        for seed, lgts, native in ((42, 0.90, 0.88), (123, 0.91, 0.89)):
            for activation, score in (("LGTS-f2-Slope", lgts), ("Native-ReLU", native)):
                rows.append({
                    "dataset": "demo", "model": "ResNet18", "activation_scope": "full_backbone",
                    "activation": activation, "activation_variant": activation,
                    "protocol": "fixed", "gpu_model": "demo-gpu", "seed": seed, "fold": 1,
                    "best_validation_f1": score, "mean_epoch_time_sec": 1.0,
                    "inference_ms_per_image": 0.1,
                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            build_aggregate_reports(path, root)
            self.assertTrue((root / "summary.csv").exists())
            self.assertTrue((root / "statistical_tests.csv").exists())
            self.assertTrue((root / "aggregate_plots" / "demo_activation_boxplot.png").exists())

    def test_q2_workbook_includes_locked_protocol_and_efficiency_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            efficiency = root / "efficiency"
            efficiency.mkdir()
            with (efficiency / "efficiency_demo.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "activation", "image_size", "forward_ms_mean"])
                writer.writeheader()
                writer.writerow({"model": "ResNet18", "activation": "LGTS-f4-Alpha", "image_size": 32, "forward_ms_mean": 1.25})
            path = root / "live_results_q2.xlsx"
            initialize_live_workbook(path)
            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                self.assertIn("Efficiency_Benchmark", workbook.sheetnames)
                self.assertIn("Protocol", workbook.sheetnames)
                self.assertEqual(workbook["Live_Dashboard"]["A1"].value, "LGTS Q2 Confirmatory Official-Test Tracker")
                self.assertEqual(workbook["Efficiency_Benchmark"]["A2"].value, "ResNet18")
                fields = [row[0] for row in workbook["Protocol"].iter_rows(min_row=2, values_only=True)]
                self.assertIn("planned_runs", fields)
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()
