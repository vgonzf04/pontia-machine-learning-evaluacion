"""Evaluacion comun para modelos de clasificacion binaria."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_NN_THRESHOLD = 0.4
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


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
    _validate_same_length(y_true, y_pred, y_score)
    _validate_binary_target(y_true)
    _validate_binary_predictions(y_pred)


def _validate_same_length(*arrays: np.ndarray) -> None:
    """Comprueba que todos los vectores contienen las mismas muestras."""
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Los vectores deben tener la misma longitud.")


def _validate_binary_target(y_true: np.ndarray) -> None:
    """Comprueba que el target incluye las dos clases del problema."""
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("y_true debe contener las clases binarias 0 y 1.")


def _validate_binary_predictions(y_pred: np.ndarray) -> None:
    """Comprueba que las predicciones solo contienen clases binarias."""
    if not set(np.unique(y_pred)).issubset({0, 1}):
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


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    model_name: str,
) -> ConfusionMatrixDisplay:
    """Crea la matriz de confusion individual de un modelo binario."""
    from matplotlib import pyplot as plt

    y_true_array = _to_one_dimension(y_true, "y_true")
    y_pred_array = _to_one_dimension(y_pred, "y_pred")

    _validate_same_length(y_true_array, y_pred_array)
    _validate_binary_target(y_true_array)
    _validate_binary_predictions(y_pred_array)

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
        display_labels=["No cancelada (0)", "Cancelada (1)"],
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=axis,
    )
    axis.set(
        title=f"Matriz de confusión - {model_name}",
        xlabel="Clase predicha",
        ylabel="Clase real",
    )
    figure.tight_layout()

    return display


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    model_name: str,
) -> RocCurveDisplay:
    """Crea la curva ROC individual usando scores de la clase positiva."""
    from matplotlib import pyplot as plt

    y_true_array = _to_one_dimension(y_true, "y_true")
    y_score_array = _to_one_dimension(y_score, "y_score")

    _validate_same_length(y_true_array, y_score_array)
    _validate_binary_target(y_true_array)

    figure, axis = plt.subplots(figsize=(7, 5))
    display = RocCurveDisplay.from_predictions(
        y_true_array,
        y_score_array,
        name=model_name,
        ax=axis,
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="grey",
        label="Clasificador aleatorio",
    )
    axis.set(
        title=f"Curva ROC - {model_name}",
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()

    return display


def _validate_model_results(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Valida y normaliza las metricas utilizadas en la comparacion."""
    if not isinstance(results, Mapping) or not results:
        raise ValueError("results debe contener al menos un modelo.")

    normalized_results: dict[str, dict[str, float]] = {}

    for model_name, metrics in results.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Cada modelo debe tener un nombre no vacio.")
        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"Las metricas de {model_name} deben estar en un diccionario."
            )

        missing_metrics = set(METRIC_LABELS).difference(metrics)
        if missing_metrics:
            missing_names = ", ".join(sorted(missing_metrics))
            raise ValueError(
                f"Faltan metricas para {model_name}: {missing_names}."
            )

        normalized_metrics: dict[str, float] = {}
        for metric_name in METRIC_LABELS:
            metric_value = metrics[metric_name]
            if isinstance(metric_value, (bool, np.bool_)):
                raise TypeError(
                    f"La metrica {metric_name} de {model_name} debe ser numerica."
                )
            try:
                numeric_value = float(metric_value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"La metrica {metric_name} de {model_name} debe ser numerica."
                ) from error

            if not np.isfinite(numeric_value) or not 0 <= numeric_value <= 1:
                raise ValueError(
                    f"La metrica {metric_name} de {model_name} debe estar entre 0 y 1."
                )
            normalized_metrics[metric_name] = numeric_value

        normalized_results[model_name] = normalized_metrics

    return normalized_results


def create_model_comparison_table(
    results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Crea una tabla comparable sin redondear las metricas originales."""
    normalized_results = _validate_model_results(results)
    rows = []

    for model_name, metrics in normalized_results.items():
        row: dict[str, Any] = {"Modelo": model_name}
        row.update(
            {
                display_name: metrics[metric_name]
                for metric_name, display_name in METRIC_LABELS.items()
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=["Modelo", *METRIC_LABELS.values()])


def plot_comparative_roc_curve(
    y_true: Any,
    scores_by_model: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, float]]:
    """Dibuja las curvas ROC disponibles sobre los mismos ejes."""
    from matplotlib import pyplot as plt

    y_true_array = _to_one_dimension(y_true, "y_true")
    _validate_binary_target(y_true_array)

    if not isinstance(scores_by_model, Mapping) or not scores_by_model:
        raise ValueError("scores_by_model debe contener al menos un modelo.")

    figure, axis = plt.subplots(figsize=(8, 6))
    auc_by_model: dict[str, float] = {}

    for model_name, scores in scores_by_model.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Cada modelo debe tener un nombre no vacio.")

        score_array = _to_one_dimension(scores, f"scores de {model_name}")
        _validate_same_length(y_true_array, score_array)

        try:
            score_array = score_array.astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Los scores de {model_name} deben ser numericos."
            ) from error
        if not np.all(np.isfinite(score_array)):
            raise ValueError(f"Los scores de {model_name} deben ser finitos.")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true_array,
            score_array,
        )
        model_auc = float(roc_auc_score(y_true_array, score_array))
        auc_by_model[model_name] = model_auc
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC = {model_auc:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="grey",
        label="Clasificador aleatorio",
    )
    axis.set(
        title="Comparacion de curvas ROC",
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()

    return figure, axis, auc_by_model


def select_best_model(
    results: Mapping[str, Mapping[str, Any]],
    primary_metric: str,
    required_models: Collection[str] | None = None,
) -> str:
    """Selecciona el mejor modelo por una metrica explicita y sin desempates ocultos."""
    normalized_results = _validate_model_results(results)

    if primary_metric not in METRIC_LABELS:
        valid_metrics = ", ".join(METRIC_LABELS)
        raise ValueError(
            f"Metrica no valida: {primary_metric}. Opciones: {valid_metrics}."
        )

    if required_models is not None:
        if isinstance(required_models, str):
            raise TypeError("required_models debe ser una coleccion de nombres.")
        missing_models = set(required_models).difference(normalized_results)
        if missing_models:
            missing_names = ", ".join(sorted(missing_models))
            raise ValueError(f"Faltan modelos requeridos: {missing_names}.")

    best_score = max(
        metrics[primary_metric] for metrics in normalized_results.values()
    )
    winners = [
        model_name
        for model_name, metrics in normalized_results.items()
        if metrics[primary_metric] == best_score
    ]

    if len(winners) > 1:
        tied_names = ", ".join(winners)
        raise ValueError(
            f"Empate en {primary_metric} entre: {tied_names}. "
            "Define un criterio de desempate."
        )

    return winners[0]
