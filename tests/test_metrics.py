import unittest

import numpy as np
import torch

from lgts_benchmark.metrics import classification_metrics, nll_from_logits


class ErrorRateTests(unittest.TestCase):
    def test_error_rate_is_one_minus_accuracy(self):
        y_true = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
        ])
        metrics, _, _ = classification_metrics(y_true, probabilities, ["zero", "one"])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["error_rate"], 0.25)
        self.assertAlmostEqual(metrics["error_rate_percent"], 25.0)
        self.assertAlmostEqual(metrics["accuracy"] + metrics["error_rate"], 1.0)

    def test_nll_from_logits_matches_cross_entropy_for_extreme_logits(self):
        logits = np.asarray([[80.0, -80.0], [-90.0, 90.0], [40.0, -40.0]], dtype=np.float32)
        y_true = np.asarray([0, 1, 1])
        expected = torch.nn.functional.cross_entropy(
            torch.from_numpy(logits), torch.from_numpy(y_true), reduction="mean",
        ).item()
        self.assertAlmostEqual(nll_from_logits(y_true, logits), expected, places=5)

    def test_top5_accuracy_is_reported(self):
        y_true = np.arange(6)
        probabilities = np.asarray([
            [0.60, 0.08, 0.08, 0.08, 0.08, 0.08],
            [0.08, 0.60, 0.08, 0.08, 0.08, 0.08],
            [0.08, 0.08, 0.60, 0.08, 0.08, 0.08],
            [0.08, 0.08, 0.08, 0.60, 0.08, 0.08],
            [0.08, 0.08, 0.08, 0.08, 0.60, 0.08],
            [0.30, 0.25, 0.20, 0.15, 0.09, 0.01],
        ])
        metrics, _, _ = classification_metrics(y_true, probabilities, [str(i) for i in range(6)])
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["top5_accuracy"], 5 / 6)


if __name__ == "__main__":
    unittest.main()
