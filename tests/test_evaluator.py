"""Pruebas unitarias del evaluador comun."""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.evaluator import (
    calculate_binary_classification_metrics,
    create_model_comparison_table,
    evaluate_nn_model,
    evaluate_sklearn_model,
    plot_comparative_roc_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    select_best_model,
)


class FakeSklearnProbabilityModel:
    """Modelo falso con las clases en orden inverso para probar su seleccion."""

    classes_ = np.array([1, 0])

    def predict(self, X_test):
        return np.array([0, 1, 1, 0])

    def predict_proba(self, X_test):
        positive_scores = np.array([0.1, 0.9, 0.8, 0.2])
        return np.column_stack((positive_scores, 1 - positive_scores))


class FakeSklearnDecisionModel:
    """Modelo falso que ofrece decision_function en vez de probabilidades."""

    classes_ = np.array([0, 1])

    def predict(self, X_test):
        return np.array([0, 1, 1, 0])

    def decision_function(self, X_test):
        return np.array([-2.0, 1.5, 0.8, -0.5])


class FakeNeuralNetwork:
    """Red falsa que reproduce la salida (n, 1) habitual de Keras."""

    def predict(self, X_test, verbose=0):
        return np.array([[0.1], [0.8], [0.4], [0.3]])


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.X_test = np.zeros((4, 2))
        self.y_test = np.array([0, 1, 1, 0])
        self.results = {
            "Modelo A": {
                "accuracy": 0.95,
                "precision": 0.75,
                "recall": 0.70,
                "f1": 0.72,
                "roc_auc": 0.80,
            },
            "Modelo B": {
                "accuracy": 0.85,
                "precision": 0.90,
                "recall": 0.88,
                "f1": 0.89,
                "roc_auc": 0.92,
            },
        }

    def tearDown(self):
        plt.close("all")

    def test_calculate_metrics_returns_expected_keys(self):
        metrics = calculate_binary_classification_metrics(
            self.y_test,
            np.array([0, 1, 1, 0]),
            np.array([0.1, 0.9, 0.8, 0.2]),
        )

        self.assertEqual(
            set(metrics),
            {"accuracy", "precision", "recall", "f1", "roc_auc"},
        )
        self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))

    def test_sklearn_uses_probability_of_class_one(self):
        metrics = evaluate_sklearn_model(
            FakeSklearnProbabilityModel(),
            self.X_test,
            self.y_test,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_sklearn_accepts_decision_function(self):
        metrics = evaluate_sklearn_model(
            FakeSklearnDecisionModel(),
            self.X_test,
            self.y_test,
        )

        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_neural_network_flattens_scores_and_applies_threshold(self):
        metrics = evaluate_nn_model(
            FakeNeuralNetwork(),
            self.X_test,
            self.y_test,
            threshold=0.4,
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_metrics_reject_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "misma longitud"):
            calculate_binary_classification_metrics(
                self.y_test,
                np.array([0, 1]),
                np.array([0.1, 0.9, 0.8, 0.2]),
            )

    def test_neural_network_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "entre 0 y 1"):
            evaluate_nn_model(
                FakeNeuralNetwork(),
                self.X_test,
                self.y_test,
                threshold=1.1,
            )

    def test_confusion_matrix_contains_expected_counts(self):
        display = plot_confusion_matrix(
            self.y_test,
            np.array([0, 1, 1, 1]),
            "Modelo de prueba",
        )

        np.testing.assert_array_equal(
            display.confusion_matrix,
            np.array([[1, 1], [0, 2]]),
        )
        self.assertEqual(
            [label.get_text() for label in display.ax_.get_xticklabels()],
            ["No cancelada (0)", "Cancelada (1)"],
        )
        self.assertIn("Modelo de prueba", display.ax_.get_title())
        self.assertEqual(display.ax_.get_xlabel(), "Clase predicha")
        self.assertEqual(display.ax_.get_ylabel(), "Clase real")

    def test_roc_curve_uses_scores_and_adds_random_baseline(self):
        display = plot_roc_curve(
            self.y_test,
            np.array([0.1, 0.9, 0.8, 0.2]),
            "Modelo de prueba",
        )

        self.assertEqual(display.roc_auc, 1.0)
        self.assertEqual(len(display.ax_.lines), 2)
        baseline = display.ax_.lines[1]
        np.testing.assert_array_equal(baseline.get_xdata(), [0, 1])
        np.testing.assert_array_equal(baseline.get_ydata(), [0, 1])
        self.assertIn("Modelo de prueba", display.ax_.get_title())
        self.assertEqual(
            display.ax_.get_xlabel(),
            "Tasa de falsos positivos",
        )
        self.assertEqual(
            display.ax_.get_ylabel(),
            "Tasa de verdaderos positivos",
        )

    def test_confusion_matrix_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "misma longitud"):
            plot_confusion_matrix(
                self.y_test,
                np.array([0, 1]),
                "Modelo de prueba",
            )

    def test_roc_curve_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "misma longitud"):
            plot_roc_curve(
                self.y_test,
                np.array([0.1, 0.9]),
                "Modelo de prueba",
            )

    def test_comparison_table_preserves_models_columns_and_values(self):
        table = create_model_comparison_table(self.results)

        self.assertEqual(
            list(table.columns),
            ["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        )
        self.assertEqual(list(table["Modelo"]), ["Modelo A", "Modelo B"])
        self.assertEqual(table.loc[0, "Accuracy"], 0.95)
        self.assertEqual(table.loc[1, "ROC-AUC"], 0.92)

    def test_comparison_rejects_missing_and_out_of_range_metrics(self):
        incomplete_results = {"Modelo A": {"accuracy": 0.8}}
        with self.assertRaisesRegex(ValueError, "Faltan metricas"):
            create_model_comparison_table(incomplete_results)

        invalid_results = {
            "Modelo A": {
                "accuracy": 1.1,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "roc_auc": 0.8,
            }
        }
        with self.assertRaisesRegex(ValueError, "entre 0 y 1"):
            create_model_comparison_table(invalid_results)

    def test_comparative_roc_contains_each_model_auc_and_baseline(self):
        figure, axis, auc_by_model = plot_comparative_roc_curve(
            self.y_test,
            {
                "Modelo A": np.array([0.1, 0.9, 0.8, 0.2]),
                "Modelo B": np.array([0.2, 0.7, 0.6, 0.3]),
            },
        )

        self.assertIs(axis.figure, figure)
        self.assertEqual(auc_by_model, {"Modelo A": 1.0, "Modelo B": 1.0})
        self.assertEqual(len(axis.lines), 3)
        self.assertIn("Modelo A (AUC = 1.000)", axis.get_legend_handles_labels()[1])
        baseline = axis.lines[-1]
        np.testing.assert_array_equal(baseline.get_xdata(), [0, 1])
        np.testing.assert_array_equal(baseline.get_ydata(), [0, 1])

    def test_comparative_roc_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "misma longitud"):
            plot_comparative_roc_curve(
                self.y_test,
                {"Modelo A": np.array([0.1, 0.9])},
            )

    def test_model_selection_uses_configured_metric(self):
        self.assertEqual(
            select_best_model(self.results, "accuracy"),
            "Modelo A",
        )
        self.assertEqual(
            select_best_model(self.results, "f1"),
            "Modelo B",
        )

    def test_model_selection_rejects_invalid_metric(self):
        with self.assertRaisesRegex(ValueError, "Metrica no valida"):
            select_best_model(self.results, "loss")

    def test_model_selection_checks_required_models(self):
        with self.assertRaisesRegex(ValueError, "Faltan modelos requeridos"):
            select_best_model(
                self.results,
                "f1",
                required_models=["Modelo A", "Modelo B", "Modelo C"],
            )

    def test_model_selection_rejects_ties(self):
        tied_results = {
            model_name: {**metrics, "f1": 0.9}
            for model_name, metrics in self.results.items()
        }

        with self.assertRaisesRegex(ValueError, "Empate en f1"):
            select_best_model(tied_results, "f1")


if __name__ == "__main__":
    unittest.main()
