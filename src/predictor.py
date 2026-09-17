import joblib
from tensorflow import keras
from data_loader import preprocessed_sklearn, preprocessed_for_prediction_nn
from pandas import DataFrame

NN_THRESHOLD = 0.4 # move it to config.py 

def sklearn_models_predict(reservation_df:DataFrame):
    X_preprocessed_sklearn = preprocessed_sklearn(reservation_df)

    lr_model = joblib.load("models/tests/logistic_regression.pkl")
    dt_model = joblib.load("models/tests/decision_tree.pkl")
    rf_model = joblib.load("models/tests/random_forest.pkl")
    xgb_model = joblib.load("models/tests/xgboost.pkl")

    y_pred_lr = lr_model.predict(X_preprocessed_sklearn)
    y_pred_dt = dt_model.predict(X_preprocessed_sklearn)
    y_pred_rf = rf_model.predict(X_preprocessed_sklearn)
    y_pred_xgb = xgb_model.predict(X_preprocessed_sklearn)

    return y_pred_lr, y_pred_dt, y_pred_rf, y_pred_xgb

def neural_network_model_predict(reservation_df:DataFrame):
    X_preprocessed_nn = preprocessed_for_prediction_nn(reservation_df)
    nn_model = keras.models.load_model("models/tests/neural_network.keras")
    y_prob_nn = nn_model.predict(X_preprocessed_nn)
    y_pred_nn = (y_prob_nn > NN_THRESHOLD).astype(int)

    return y_prob_nn, y_pred_nn