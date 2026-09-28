"""Persistencia de tablas y figuras generadas durante la evaluacion."""

from pathlib import Path
from typing import Any

import pandas as pd


def _prepare_output_path(output_path: str | Path, expected_suffix: str) -> Path:
    """Valida la extension y crea el directorio de salida si es necesario."""
    path = Path(output_path)

    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"La ruta de salida debe terminar en {expected_suffix}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Guarda una tabla como CSV sin incluir el indice del DataFrame."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table debe ser un DataFrame de pandas.")

    path = _prepare_output_path(output_path, ".csv")
    table.to_csv(path, index=False)
    return path


def save_figure(figure: Any, output_path: str | Path) -> Path:
    """Guarda una figura como PNG y la cierra para liberar memoria."""
    if not hasattr(figure, "savefig"):
        raise TypeError("figure debe ser una figura de Matplotlib.")

    path = _prepare_output_path(output_path, ".png")
    figure.savefig(path, dpi=150, bbox_inches="tight")

    from matplotlib import pyplot as plt

    plt.close(figure)
    return path
