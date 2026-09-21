import joblib
from tensorflow import keras
from data_loader import preprocessed_for_sklearn_prediction, preprocessed_for_nn_prediction
import pandas as pd
from pathlib import Path

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
    X_preprocessed_sklearn = preprocessed_for_sklearn_prediction()

    # cargamos el modelo
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")

    # calculamos precciones
    y_pred = best_model.predict(X_preprocessed_sklearn)

    # devolvemos predicciones
    return y_pred

def neural_network_model_predict():
    # preprocesamiento de datos para red neuronal
    X_preprocessed_nn = preprocessed_for_nn_prediction()

    # cargamos el modelo de red neuronal
    best_model = keras.models.load_model(MODELS_DIR / "best_model.keras")

    # predecimos, devuelve probabilidades
    y_prob_nn = best_model.predict(X_preprocessed_nn)
    # con las probabilidades y el threshold elegido durante el experimentacion calculamos prediciones
    y_pred_nn = (y_prob_nn > NN_THRESHOLD).astype(int)

    # devolvemos predicciones
    return y_pred_nn


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

    else:
        # caso de red neuronal
        y_pred = neural_network_model_predict()


    # print para ver predicciones 

if __name__ == "__main__":
    main()