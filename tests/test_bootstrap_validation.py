import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heart_disease.bootstrap_validation import (
    bootstrap_optimism_correction,
    optimism_summary_table,
)
from heart_disease.data import (
    create_target,
    load_data,
    split_features_target,
    split_train_test,
)
from heart_disease.evaluation import calibration_parameters
from heart_disease.models import build_logistic_regression
from heart_disease.preprocessing import build_preprocessor, remove_missing_rows

METRICS = ("auc", "brier", "slope", "intercept")


class BootstrapValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = create_target(load_data())
        features, target = split_features_target(data)
        (
            cls.x_train,
            _,
            cls.y_train,
            _,
        ) = split_train_test(features, target)
        cls.x_train, cls.y_train = remove_missing_rows(cls.x_train, cls.y_train)
        cls.result = bootstrap_optimism_correction(
            cls._fit_model(),
            cls.x_train,
            cls.y_train,
            n_iterations=50,
            random_state=42,
        )

    @classmethod
    def _fit_model(cls):
        model = build_logistic_regression()
        model.fit(cls.x_train, cls.y_train)
        return model

    def test_result_contains_expected_keys(self):
        for key in (
            "apparent",
            "mean_optimism",
            "corrected",
            "corrected_ci",
            "records",
            "n_resamples",
        ):
            self.assertIn(key, self.result)
        for metric in METRICS:
            self.assertIn(metric, self.result["apparent"])
            self.assertIn(metric, self.result["mean_optimism"])
            self.assertIn(metric, self.result["corrected"])
            self.assertIn(metric, self.result["corrected_ci"])

    def test_n_resamples_matches_records(self):
        self.assertEqual(
            self.result["n_resamples"],
            len(self.result["records"]),
        )
        self.assertGreater(self.result["n_resamples"], 0)

    def test_corrected_equals_apparent_minus_optimism(self):
        for metric in METRICS:
            self.assertAlmostEqual(
                self.result["corrected"][metric],
                self.result["apparent"][metric]
                - self.result["mean_optimism"][metric],
                places=10,
            )

    def test_mean_optimism_matches_records(self):
        pairs = {
            "auc": ("auc_apparent", "auc_test"),
            "brier": ("brier_apparent", "brier_test"),
            "slope": ("slope_apparent", "slope_test"),
            "intercept": ("intercept_apparent", "intercept_test"),
        }
        for metric, (apparent_column, test_column) in pairs.items():
            expected = (
                self.result["records"][apparent_column]
                - self.result["records"][test_column]
            ).mean()
            self.assertAlmostEqual(
                self.result["mean_optimism"][metric],
                expected,
                places=10,
            )

    def test_apparent_slope_is_one_for_near_mle_fit(self):
        # 无惩罚（接近 MLE）逻辑回归的表观校准斜率恒为 1、截距恒为 0；
        # 项目默认 C=1 带 L2 惩罚，表观斜率会大于 1，因此这里用 C=1e6 验证
        model = Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=1000, C=1e6)),
            ]
        )
        model.fit(self.x_train, self.y_train)
        probability = model.predict_proba(self.x_train)[:, 1]
        intercept, slope = calibration_parameters(self.y_train, probability)
        self.assertAlmostEqual(slope, 1.0, delta=1e-2)
        self.assertAlmostEqual(intercept, 0.0, delta=1e-2)

    def test_corrected_slope_below_apparent(self):
        # 期望方向：存在过拟合时，校正后斜率低于表观斜率
        self.assertLess(
            self.result["corrected"]["slope"],
            self.result["apparent"]["slope"],
        )

    def test_all_values_are_finite(self):
        for metric in METRICS:
            self.assertTrue(np.isfinite(self.result["apparent"][metric]))
            self.assertTrue(np.isfinite(self.result["mean_optimism"][metric]))
            self.assertTrue(np.isfinite(self.result["corrected"][metric]))

    def test_random_state_is_deterministic(self):
        second = bootstrap_optimism_correction(
            self._fit_model(),
            self.x_train,
            self.y_train,
            n_iterations=50,
            random_state=42,
        )
        for metric in METRICS:
            self.assertAlmostEqual(
                self.result["corrected"][metric],
                second["corrected"][metric],
                places=10,
            )
        np.testing.assert_allclose(
            self.result["records"].to_numpy(),
            second["records"].to_numpy(),
        )

    def test_summary_table_shape(self):
        table = optimism_summary_table(self.result)
        self.assertEqual(len(table), len(METRICS))
        self.assertEqual(
            list(table.columns),
            ["metric", "apparent", "mean_optimism", "corrected", "ci_lower", "ci_upper"],
        )


if __name__ == "__main__":
    unittest.main()
