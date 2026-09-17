import joblib
from tensorflow import keras
from data_loader import preprocessed_sklearn, preprocessed_for_prediction_nn
from pandas import DataFrame
from pathlib import Path

# threshold para marcar a partir de que prob es 0 o 1
NN_THRESHOLD = 0.4 # move it to config.py
# directorio de modelos guardados y entrenados
MODELS_DIR = Path("models/tests")

def sklearn_models_predict(reservation_df:DataFrame):
    X_preprocessed_sklearn = preprocessed_sklearn(reservation_df)

    lr_model = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    dt_model = joblib.load(MODELS_DIR / "decision_tree.pkl")
    rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")

    y_pred_lr = lr_model.predict(X_preprocessed_sklearn)
    y_pred_dt = dt_model.predict(X_preprocessed_sklearn)
    y_pred_rf = rf_model.predict(X_preprocessed_sklearn)
    y_pred_xgb = xgb_model.predict(X_preprocessed_sklearn)

    return y_pred_lr, y_pred_dt, y_pred_rf, y_pred_xgb

def neural_network_model_predict(reservation_df:DataFrame):
    X_preprocessed_nn = preprocessed_for_prediction_nn(reservation_df)
    nn_model = keras.models.load_model(MODELS_DIR / "neural_network.keras")
    y_prob_nn = nn_model.predict(X_preprocessed_nn)
    y_pred_nn = (y_prob_nn > NN_THRESHOLD).astype(int)

    return y_prob_nn, y_pred_nn