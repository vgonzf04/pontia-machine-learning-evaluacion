"""Pruebas unitarias para las fabricas de modelos."""

import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models import create_random_forest_model


class RandomForestModelTests(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array(
            [
                [0.0, 0.1],
                [0.1, 0.2],
                [0.2, 0.0],
                [0.3, 0.1],
                [0.7, 0.8],
                [0.8, 0.9],
                [0.9, 0.7],
                [1.0, 0.9],
            ]
        )
        self.y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def test_factory_returns_expected_random_forest_configuration(self):
        model = create_random_forest_model()

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_jobs, -1)
        self.assertIsNone(model.class_weight)

    def test_model_can_fit_and_produce_binary_probabilities(self):
        model = create_random_forest_model()
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_train)
        probabilities = model.predict_proba(self.X_train)

        self.assertEqual(predictions.shape, (len(self.X_train),))
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertEqual(probabilities.shape, (len(self.X_train), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_random_state_makes_training_reproducible(self):
        first_model = create_random_forest_model()
        second_model = create_random_forest_model()

        first_model.fit(self.X_train, self.y_train)
        second_model.fit(self.X_train, self.y_train)

        np.testing.assert_allclose(
            first_model.predict_proba(self.X_train),
            second_model.predict_proba(self.X_train),
        )


if __name__ == "__main__":
    unittest.main()
