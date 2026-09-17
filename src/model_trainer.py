# archivo orquestador funciona como main, funciones 
from models import create_logistic_regression_model, create_decision_tree_model, create_random_forest_model, create_xgboost_model, create_neural_network_model
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import joblib

def sklearn_model_train(X_train_preprocessed, y_train):
    lr_model = create_logistic_regression_model()
    dt_model = create_decision_tree_model()
    rf_model = create_random_forest_model()
    xgb_model = create_xgboost_model()

    
    lr_model.fit(X_train_preprocessed, y_train)
    dt_model.fit(X_train_preprocessed, y_train)
    rf_model.fit(X_train_preprocessed, y_train)
    xgb_model.fit(X_train_preprocessed, y_train)

    return lr_model, dt_model, rf_model, xgb_model

    

def neural_network_model_train(X_train_preprocessed_nn, y_train):
    nn_model = create_neural_network_model(X_train_preprocessed_nn)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    history = nn_model.fit(X_train_preprocessed_nn, y_train, batch_size=32, epochs=200, callbacks=[early_stopping], validation_split=.2)

    return history, nn_model

def save_trained_models(lr_model, dt_model, rf_model, xgb_model, nn_model):
    MODELS_DIR = Path("models/tests")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(dt_model, MODELS_DIR / "decision_tree.pkl")
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(nn_model, MODELS_DIR / "neural_network.keras")


def main():
#   pedir datos
    X_train_preprocessed, y_train = get_preprocessed_data()
    X_train_preprocessed_nn = get_preprocessed_data_for_nn()

#   pedir modelos
#   entrenarlos
    lr_model, dt_model, rf_model, xgb_model = sklearn_model_train(X_train_preprocessed, y_train)
    history, nn_model = neural_network_model_train(X_train_preprocessed_nn, y_train)
    
#   guardarlos
    save_trained_models(lr_model, dt_model, rf_model, xgb_model, nn_model)

#   pedir a evaluator que los evalúe


#   comparar resultados
    

#   elegir el mejor


#   guardar best_model


if __name__ == "__main__":
    main()