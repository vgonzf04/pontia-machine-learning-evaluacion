"""Pruebas unitarias para la persistencia de resultados de evaluacion."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.output_manager import save_figure, save_table


class OutputManagerTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_save_table_creates_directories_and_csv_without_index(self):
        table = pd.DataFrame(
            {
                "Modelo": ["Random Forest", "XGBoost"],
                "F1": [0.81, 0.83],
            }
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory)
                / "nested"
                / "model_comparison.csv"
            )
            saved_path = save_table(table, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(saved_path.is_file())
            pd.testing.assert_frame_equal(pd.read_csv(saved_path), table)

    def test_save_figure_creates_png_and_closes_figure(self):
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        figure_number = figure.number

        with TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory)
                / "nested"
                / "roc_curve.png"
            )
            saved_path = save_figure(figure, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(saved_path.is_file())
            self.assertGreater(saved_path.stat().st_size, 0)
            self.assertFalse(plt.fignum_exists(figure_number))

    def test_save_table_rejects_invalid_object_or_extension(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            with self.assertRaisesRegex(TypeError, "DataFrame"):
                save_table([], temporary_path / "table.csv")

            with self.assertRaisesRegex(ValueError, "terminar en .csv"):
                save_table(pd.DataFrame(), temporary_path / "table.txt")

    def test_save_figure_rejects_invalid_object_or_extension(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            with self.assertRaisesRegex(TypeError, "Matplotlib"):
                save_figure(object(), temporary_path / "figure.png")

            figure = plt.figure()
            with self.assertRaisesRegex(ValueError, "terminar en .png"):
                save_figure(figure, temporary_path / "figure.svg")


if __name__ == "__main__":
    unittest.main()
