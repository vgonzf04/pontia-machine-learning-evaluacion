import joblib
from tensorflow import keras
from data_loader import preprocessed_for_tree_prediction, preprocessed_and_scaled_for_prediction
import pandas as pd
from pathlib import Path
import json


# threshold para marcar a partir de que prob es 0 o 1
NN_THRESHOLD = 0.4 # move it to config.py
# directorio de modelos guardados y entrenados
MODELS_DIR = Path("models")
# directorio de info de mejor modelo
METADATA_PATH = Path("models/best_model_metadata.json")
# directorio de datos para predecir
DATA_PATH = Path("data/raw/data_predict.csv")


def sklearn_model_predict():
    # preprocesamiento de datos para modelos clasicos
    X_preprocessed_sklearn = preprocessed_for_tree_prediction()

    # cargamos el modelo
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")

    # calculamos precciones
    y_pred = best_model.predict(X_preprocessed_sklearn)

    # devolvemos predicciones
    return y_pred

def lr_model_predict():
    # preprocesamiento de datos para modelos clasicos
    X_preprocessed_scaled = preprocessed_and_scaled_for_prediction()

    # cargamos el modelo
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")

    # calculamos precciones
    y_pred = best_model.predict(X_preprocessed_scaled)

    # devolvemos predicciones
    return y_pred

def neural_network_model_predict():
    # preprocesamiento de datos para red neuronal
    X_preprocessed_scaled = preprocessed_and_scaled_for_prediction()

    # cargamos el modelo de red neuronal
    best_model = keras.models.load_model(MODELS_DIR / "best_model.keras")

    # predecimos, devuelve probabilidades
    y_prob_nn = best_model.predict(X_preprocessed_scaled)
    # con las probabilidades y el threshold elegido durante el experimentacion calculamos prediciones
    y_pred_nn = (y_prob_nn > NN_THRESHOLD).astype(int)

    # devolvemos predicciones
    return y_pred_nn

def print_predictions(y_pred):
    print("\n" + "=" * 50)
    print("         RESULTADOS DE LAS PREDICCIONES")
    print("=" * 50)

    # aplanamos por si viene con forma (n, 1), como suele ocurrir en Keras
    predictions = y_pred.flatten()

    for i, prediction in enumerate(predictions, start=1):
        status = "CANCELADA" if prediction == 1 else "NO CANCELADA"

        print(f"Reserva {i:>3} -> {status}")

    print("=" * 50)
    print(f"Total de predicciones: {len(predictions)}")
    print("=" * 50 + "\n")


def main():
    
    # cargamos info del mejor modelo
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    # guardamos tipo de modelo 
    best_model_type = metadata["best_model_type"]

    # comparamos si mejor modelo es tipo clasico o red neuronal
    if best_model_type == "sklearn":
        # caso de modelo clasico
        y_pred = sklearn_model_predict()

    elif best_model_type == "logistic_regression":
        y_pred = lr_model_predict()

    else:
        # caso de red neuronal
        y_pred = neural_network_model_predict()

    # print para ver predicciones 
    print_predictions(y_pred)

if __name__ == "__main__":
    main()