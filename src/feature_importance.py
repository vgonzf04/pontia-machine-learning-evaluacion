"""Utilidades genericas para interpretar importancias de variables."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _validate_feature_names(feature_names: Iterable[str]) -> list[str]:
    """Valida y materializa los nombres en el orden del modelo."""
    if isinstance(feature_names, (str, bytes)):
        raise TypeError("feature_names debe ser una coleccion de nombres.")

    try:
        names = list(feature_names)
    except TypeError as error:
        raise TypeError(
            "feature_names debe ser una coleccion de nombres."
        ) from error

    if not names:
        raise ValueError("feature_names no puede estar vacio.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Cada variable debe tener un nombre de texto no vacio.")
    if len(set(names)) != len(names):
        raise ValueError("Los nombres de variables no pueden estar duplicados.")

    return names


def _get_feature_importances(model: Any) -> np.ndarray:
    """Obtiene un vector valido desde un modelo compatible y entrenado."""
    try:
        raw_importances = model.feature_importances_
    except AttributeError as error:
        raise ValueError(
            "El modelo debe estar entrenado y exponer feature_importances_."
        ) from error

    try:
        importances = np.asarray(raw_importances, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("Las importancias deben ser numericas.") from error

    if importances.ndim != 1 or importances.size == 0:
        raise ValueError("feature_importances_ debe ser un vector no vacio.")
    if not np.all(np.isfinite(importances)):
        raise ValueError("Las importancias deben ser finitas.")
    if np.any(importances < 0):
        raise ValueError("Las importancias no pueden ser negativas.")

    return importances


def create_feature_importance_table(
    model: Any,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Asocia variables e importancias y las ordena de mayor a menor."""
    names = _validate_feature_names(feature_names)
    importances = _get_feature_importances(model)

    if len(names) != len(importances):
        raise ValueError(
            "Debe existir un nombre por cada importancia del modelo."
        )

    table = pd.DataFrame(
        {
            "feature": names,
            "importance": importances,
        }
    )
    return table.sort_values(
        "importance",
        ascending=False,
        kind="mergesort",
        ignore_index=True,
    )


def plot_feature_importance(
    model: Any,
    feature_names: Iterable[str],
    model_name: str,
    top_n: int = 20,
) -> Any:
    """Representa las variables mas importantes de un modelo entrenado."""
    from matplotlib import pyplot as plt

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name debe ser un texto no vacio.")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(
        top_n,
        (int, np.integer),
    ):
        raise TypeError("top_n debe ser un numero entero.")
    if top_n <= 0:
        raise ValueError("top_n debe ser mayor que cero.")

    importance_table = create_feature_importance_table(model, feature_names)
    selected_features = importance_table.head(top_n).iloc[::-1]
    figure_height = max(4.0, 0.4 * len(selected_features) + 1.5)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    axis.barh(
        selected_features["feature"],
        selected_features["importance"],
        color="steelblue",
    )
    axis.set(
        title=f"Importancia de variables - {model_name.strip()}",
        xlabel="Importancia",
        ylabel="Variable",
    )
    figure.tight_layout()

    return axis
