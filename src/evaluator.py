"""Evaluacion comun para modelos de clasificacion binaria."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_NN_THRESHOLD = 0.4


def _to_one_dimension(values: Any, value_name: str) -> np.ndarray:
    """Convierte predicciones o etiquetas a un vector de una dimension."""
    array = np.asarray(values).reshape(-1)

    if array.size == 0:
        raise ValueError(f"{value_name} no puede estar vacio.")

    return array


def _validate_binary_classification_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Comprueba formas y valores necesarios para evaluar el problema."""
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError(
            "y_true, y_pred e y_score deben tener la misma longitud."
        )

    true_classes = set(np.unique(y_true))
    if true_classes != {0, 1}:
        raise ValueError("y_true debe contener las clases binarias 0 y 1.")

    predicted_classes = set(np.unique(y_pred))
    if not predicted_classes.issubset({0, 1}):
        raise ValueError("y_pred solo puede contener las clases 0 y 1.")


def calculate_binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
) -> dict[str, float]:
    """Calcula las metricas comunes usando scores para ROC-AUC.

    Args:
        y_true: Etiquetas reales de la muestra de test.
        y_pred: Clases 0/1 predichas por el modelo.
        y_score: Probabilidad o score asignado a la clase positiva 1.

    Returns:
        Diccionario con accuracy, precision, recall, F1 y ROC-AUC.
    """
    y_true_array = _to_one_dimension(y_true, "y_true")
    y_pred_array = _to_one_dimension(y_pred, "y_pred")
    y_score_array = _to_one_dimension(y_score, "y_score")

    _validate_binary_classification_data(
        y_true_array,
        y_pred_array,
        y_score_array,
    )

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, y_score_array)),
    }


def _positive_class_index(model: Any, number_of_columns: int) -> int:
    """Localiza la columna que representa la clase positiva 1."""
    classes = getattr(model, "classes_", None)

    if classes is None:
        if number_of_columns == 2:
            return 1
        raise ValueError(
            "No se puede identificar la clase positiva en los scores."
        )

    positive_positions = np.flatnonzero(np.asarray(classes) == 1)
    if len(positive_positions) != 1:
        raise ValueError("El modelo debe incluir exactamente una clase positiva 1.")

    return int(positive_positions[0])


def _get_sklearn_positive_scores(model: Any, X_test: Any) -> np.ndarray:
    """Obtiene probabilidades o scores sklearn para la clase positiva 1."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X_test))

        if probabilities.ndim != 2:
            raise ValueError("predict_proba debe devolver una matriz de dos dimensiones.")

        positive_index = _positive_class_index(model, probabilities.shape[1])
        return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(X_test))

        if decision_scores.ndim == 1:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = np.asarray(classes)
                if len(classes) != 2 or 1 not in classes:
                    raise ValueError(
                        "decision_function requiere las clases binarias 0 y 1."
                    )
                if classes[1] != 1:
                    decision_scores = -decision_scores
            return decision_scores

        if decision_scores.ndim == 2:
            positive_index = _positive_class_index(
                model,
                decision_scores.shape[1],
            )
            return decision_scores[:, positive_index]

        raise ValueError(
            "decision_function debe devolver un vector o una matriz."
        )

    raise TypeError(
        "El modelo sklearn debe implementar predict_proba o decision_function."
    )


def evaluate_sklearn_model(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:
    """Evalua un modelo sklearn sobre el conjunto comun de test."""
    y_pred = model.predict(X_test)
    y_score = _get_sklearn_positive_scores(model, X_test)

    return calculate_binary_classification_metrics(y_test, y_pred, y_score)


def evaluate_nn_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    history: Any = None,
    threshold: float = DEFAULT_NN_THRESHOLD,
) -> dict[str, float]:
    """Evalua una red Keras usando su probabilidad para la clase positiva."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")

    # Se conserva por compatibilidad con model_trainer.py. Las curvas de
    # aprendizaje se incorporaran en el paso dedicado a visualizaciones.
    _ = history

    y_score = _to_one_dimension(
        model.predict(X_test, verbose=0),
        "predicciones de la red neuronal",
    )
    y_pred = (y_score > threshold).astype(int)

    return calculate_binary_classification_metrics(y_test, y_pred, y_score)
