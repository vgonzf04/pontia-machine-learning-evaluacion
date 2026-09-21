from models import create_logistic_regression_model, create_decision_tree_model, create_random_forest_model, create_xgboost_model, create_neural_network_model
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import joblib
from data_loader import get_preprocessed_data, get_preprocessed_data_for_nn
from evaluator import evaluate_sklearn_model, evaluate_nn_model
import json

# rutas para guardar modelos
MODELS_DIR = Path("models/tests")
BEST_MODEL_DIR = Path("models")
# ruta para guardar info del mejor modelo
METADATA_PATH = Path("models/best_model_metadata.json")

# funcion para crear (desde models.py) y entrenar modelos clasicos, reciben X_train preprocesado e y_train
def create_train_sklearn_models(X_train_preprocessed, y_train):
    # creacion modelo regresion logistica
    lr_model = create_logistic_regression_model()
    # creacion modelo arbol decision logistica
    dt_model = create_decision_tree_model()
    # creacion modelo random forest
    rf_model = create_random_forest_model()
    # creacion modelo gradient boosting (xgboost)
    xgb_model = create_xgboost_model()

    # entrenamiento de cada modelo clasico con los mismos datos
    lr_model.fit(X_train_preprocessed, y_train)
    dt_model.fit(X_train_preprocessed, y_train)
    rf_model.fit(X_train_preprocessed, y_train)
    xgb_model.fit(X_train_preprocessed, y_train)

    # devuelve modelos entrenados
    return lr_model, dt_model, rf_model, xgb_model

    
# funcion para crear (desde models.py) y entrenar red neuronal
def create_train_neural_network_model(X_train_preprocessed_nn, y_train):
    # creacion de la red neuronal con capas y neuronas optimas
    nn_model = create_neural_network_model(X_train_preprocessed_nn)
    # funcion para que no se ejecuten mas epocas de la cuenta
    early_stopping = EarlyStopping(
        # se fija en validacion de perdida
        monitor="val_loss",
        # si no mejora en 3 epocas para
        patience=3,
        # de todas las epocas ejecutadas coge los pesos con mejores metricas
        restore_best_weights=True
    )
    # entrenamiento de la red, recibe x_train preprocesado para una red neuronal, procesa datos en bloques de 32, max 200 epocas, 20% de datos para validacion
    history = nn_model.fit(X_train_preprocessed_nn, y_train, batch_size=32, epochs=200, callbacks=[early_stopping], validation_split=.2)

    # devuelve hsitory para display de curva de aprendizaje y modelo entrenado
    return history, nn_model

# funcion para guardar los modelos entrenados 
def save_trained_models(lr_model, dt_model, rf_model, xgb_model, nn_model):
    # parents=True -> si no existe directorio lo crea; exist_ok=True -> si existe no da error
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # guarda cada modelo clasico con joblib
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(dt_model, MODELS_DIR / "decision_tree.pkl")
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    # guarda red neuronal con funcion especifica de keras
    nn_model.save(MODELS_DIR / "neural_network.keras")

# funcion para comparar modelos
def compare_models(results):
    # guarda el nombre del mejor modelo en funcion de f1-score
    best_model_name = max(results, key=lambda model_name: results[model_name]["f1"])
    # print de mejor modelo y su f1-score
    print(f"Best model: {best_model_name}, f1-score: {results[best_model_name]["f1"]}")

    # devuelve el nombre del mejor modelo
    return best_model_name


def main():
    # guardamos datos preprocesados (desde loader_data.py)
    X_train_preprocessed, X_test_preprocessed, y_train, y_test = get_preprocessed_data()
    # datos con preprocesamiento especial para red neuronal (desde loader_data.py)
    X_train_preprocessed_nn, X_test_preprocessed_nn, y_train_nn, y_test_nn = get_preprocessed_data_for_nn()

    # llamamos a funcion de crear y entrenar modelos clasicos
    lr_model, dt_model, rf_model, xgb_model = create_train_sklearn_models(X_train_preprocessed, y_train)
    # llamamos a funcion de crear y entrenar red neuronal
    history, nn_model = create_train_neural_network_model(X_train_preprocessed_nn, y_train_nn)
    
    # llamamos a funcion de guardado de modelos
    save_trained_models(lr_model, dt_model, rf_model, xgb_model, nn_model)

    # evaluamos modelos (desde evaluator.py) y lo guardamos en un diccionario conjunto
    results = {}
    results["logistic_regression"] = evaluate_sklearn_model(lr_model, X_test_preprocessed, y_test)
    results["decision_tree"] = evaluate_sklearn_model(dt_model, X_test_preprocessed, y_test)
    results["random_forest"] = evaluate_sklearn_model(rf_model, X_test_preprocessed, y_test)
    results["xgboost"] = evaluate_sklearn_model(xgb_model, X_test_preprocessed, y_test)
    results["neural_network"] = evaluate_nn_model(nn_model, X_test_preprocessed_nn, y_test_nn, history) # threshold > 0.4

    # diccionario nombre_modelo: modelos
    trained_models = {
        "logistic_regression": lr_model,
        "decision_tree": dt_model,
        "random_forest": rf_model,
        "xgboost": xgb_model,
        "neural_network": nn_model 
    }

    # llamamos a funcion de comapracion
    best_model_name = compare_models(results)
    # sacamos mejor modelo con diccionario
    best_model = trained_models[best_model_name]
    
    # ruta de mejor modelo 
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # vemos que tipo de modelo es para guardado
    if best_model_name == "neural_network":
        # caso de red neuronal
        nn_model.save(BEST_MODEL_DIR / "best_model.keras")

        # info mejor modelo
        metadata = {
            "best_model_name": best_model_name,
            "best_model_type": "keras"
        }

        # guardar info mejor modelo 
        with open(METADATA_PATH, "w") as file:
            json.dump(metadata, file, indent=4)

    else:
        # caso de modelo clasico
        joblib.dump(best_model, BEST_MODEL_DIR / "best_model.pkl")

        # info mejor modelo
        metadata = {
            "best_model_name": best_model_name,
            "best_model_type": "sklearn"
        }

        # guardar info mejor modelo 
        with open(METADATA_PATH, "w") as file:
            json.dump(metadata, file, indent=4)

if __name__ == "__main__":
    main()