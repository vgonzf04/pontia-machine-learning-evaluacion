"""Pruebas unitarias para las utilidades de feature importance."""

import unittest

import matplotlib
import numpy as np
from sklearn.datasets import make_classification

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.feature_importance import (
    create_feature_importance_table,
    plot_feature_importance,
)
from src.models import create_random_forest_model, create_xgboost_model


class FakeFeatureImportanceModel:
    def __init__(self, importances):
        self.feature_importances_ = importances


class FeatureImportanceTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeFeatureImportanceModel([0.2, 0.6, 0.2])
        self.feature_names = ["lead_time", "adr", "country"]

    def tearDown(self):
        plt.close("all")

    def test_table_associates_and_sorts_importances_stably(self):
        table = create_feature_importance_table(
            self.model,
            self.feature_names,
        )

        self.assertEqual(list(table.columns), ["feature", "importance"])
        self.assertEqual(
            list(table["feature"]),
            ["adr", "lead_time", "country"],
        )
        np.testing.assert_allclose(table["importance"], [0.6, 0.2, 0.2])

    def test_plot_limits_features_and_places_most_important_on_top(self):
        axis = plot_feature_importance(
            self.model,
            self.feature_names,
            model_name="Random Forest",
            top_n=2,
        )

        self.assertEqual(len(axis.patches), 2)
        self.assertEqual(
            [label.get_text() for label in axis.get_yticklabels()],
            ["lead_time", "adr"],
        )
        self.assertIn("Random Forest", axis.get_title())
        self.assertEqual(axis.get_xlabel(), "Importancia")
        self.assertEqual(axis.get_ylabel(), "Variable")

    def test_project_tree_models_are_supported_with_synthetic_data(self):
        X_train, y_train = make_classification(
            n_samples=60,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]

        for factory in (create_random_forest_model, create_xgboost_model):
            with self.subTest(factory=factory.__name__):
                model = factory()
                model.fit(X_train, y_train)
                table = create_feature_importance_table(model, feature_names)

                self.assertEqual(len(table), len(feature_names))
                self.assertAlmostEqual(table["importance"].sum(), 1.0)

    def test_table_rejects_model_without_feature_importances(self):
        with self.assertRaisesRegex(ValueError, "feature_importances_"):
            create_feature_importance_table(object(), self.feature_names)

    def test_table_rejects_wrong_number_of_feature_names(self):
        with self.assertRaisesRegex(ValueError, "un nombre por cada importancia"):
            create_feature_importance_table(self.model, ["feature_1"])

    def test_table_rejects_invalid_feature_names(self):
        invalid_names = (
            ["feature_1", "feature_1", "feature_3"],
            ["feature_1", "", "feature_3"],
        )

        for feature_names in invalid_names:
            with self.subTest(feature_names=feature_names):
                with self.assertRaises(ValueError):
                    create_feature_importance_table(self.model, feature_names)

    def test_table_rejects_invalid_importances(self):
        invalid_importances = (
            [0.2, -0.1, 0.9],
            [0.2, np.inf, 0.8],
            [0.2, "invalid", 0.8],
        )

        for importances in invalid_importances:
            with self.subTest(importances=importances):
                with self.assertRaises((TypeError, ValueError)):
                    create_feature_importance_table(
                        FakeFeatureImportanceModel(importances),
                        self.feature_names,
                    )

    def test_plot_rejects_invalid_top_n(self):
        for top_n in (0, -1, 1.5, True):
            with self.subTest(top_n=top_n):
                with self.assertRaises((TypeError, ValueError)):
                    plot_feature_importance(
                        self.model,
                        self.feature_names,
                        model_name="Random Forest",
                        top_n=top_n,
                    )


if __name__ == "__main__":
    unittest.main()
