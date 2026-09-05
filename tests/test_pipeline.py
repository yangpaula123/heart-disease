import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heart_disease.data import (
    create_target,
    load_data,
    split_features_target,
    split_train_test,
)
from heart_disease.models import build_logistic_regression
from heart_disease.preprocessing import remove_missing_rows


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = create_target(load_data())
        features, target = split_features_target(data)
        (
            cls.x_train,
            cls.x_test,
            cls.y_train,
            cls.y_test,
        ) = split_train_test(features, target)
        cls.x_train, cls.y_train = remove_missing_rows(cls.x_train, cls.y_train)
        cls.x_test, cls.y_test = remove_missing_rows(cls.x_test, cls.y_test)

    def test_target_is_binary(self):
        self.assertTrue(set(self.y_train.unique()).issubset({0, 1}))

    def test_features_and_target_are_aligned(self):
        self.assertEqual(len(self.x_train), len(self.y_train))
        self.assertEqual(len(self.x_test), len(self.y_test))
        self.assertFalse(self.x_train.isna().any().any())
        self.assertFalse(self.x_test.isna().any().any())

    def test_stratified_split_is_close(self):
        self.assertLess(abs(self.y_train.mean() - self.y_test.mean()), 0.1)

    def test_logistic_pipeline_predicts_probabilities(self):
        model = build_logistic_regression()
        model.fit(self.x_train, self.y_train)
        probability = model.predict_proba(self.x_test)[:, 1]
        self.assertEqual(len(probability), len(self.y_test))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
