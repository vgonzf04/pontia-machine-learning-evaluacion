"""Pruebas unitarias del evaluador comun."""

import unittest

import numpy as np

from src.evaluator import (
    calculate_binary_classification_metrics,
    evaluate_nn_model,
    evaluate_sklearn_model,
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


if __name__ == "__main__":
    unittest.main()
