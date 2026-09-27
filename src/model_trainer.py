from models import create_logistic_regression_model, create_decision_tree_model, create_random_forest_model, create_xgboost_model, create_neural_network_model
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import joblib
from data_loader import get_preprocessed_data
import json
from evaluator import (
    evaluate_sklearn_model,
    evaluate_nn_model,
    create_model_comparison_table,
    select_best_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_comparative_roc_curve,
)
from feature_importance import (
    create_feature_importance_table,
    plot_feature_importance
)
from predictor import NN_THRESHOLD

# rutas para guardar modelos
MODELS_DIR = Path("models/tests")
BEST_MODEL_DIR = Path("models")
# ruta para guardar info del mejor modelo
METADATA_PATH = Path("models/best_model_metadata.json")

# funcion para crear modelos (desde models.py) y entrenarlos
def create_train_sklearn_models(X_train_preprocessed, y_train):
    # creacion modelo arbol decision logistica
    dt_model = create_decision_tree_model()
    # creacion modelo random forest
    rf_model = create_random_forest_model()
    # creacion modelo gradient boosting (xgboost)
    xgb_model = create_xgboost_model()

    # entrenamiento de cada modelo clasico con los mismos datos
    dt_model.fit(X_train_preprocessed, y_train)
    rf_model.fit(X_train_preprocessed, y_train)
    xgb_model.fit(X_train_preprocessed, y_train)

    # devuelve modelos entrenados
    return dt_model, rf_model, xgb_model

    
# funcion para crear modelos(desde models.py) y entrenarlos
def create_train_lr_and_nn_model(X_train_scaled_preprocessed, y_train):
    # creacion modelo regresion logistica
    lr_model = create_logistic_regression_model()
    # entramiento del modelo de regresion logistica
    lr_model.fit(X_train_scaled_preprocessed, y_train)

    # creacion de la red neuronal con capas y neuronas optimas
    nn_model = create_neural_network_model(X_train_scaled_preprocessed)
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
    history = nn_model.fit(X_train_scaled_preprocessed, y_train, batch_size=32, epochs=200, callbacks=[early_stopping], validation_split=.2)

    # devuelve hsitory para display de curva de aprendizaje y modelo entrenado
    return lr_model, history, nn_model

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


def main():
    # guardamos datos preprocesados (desde loader_data.py)
    X_train_preprocessed, X_test_preprocessed, X_train_scaled, X_test_scaled, y_train, y_test, feature_names = get_preprocessed_data()
    
    # llamamos a funcion de crear y entrenar modelos clasicos
    dt_model, rf_model, xgb_model = create_train_sklearn_models(X_train_preprocessed, y_train)
    # llamamos a funcion de crear y entrenar red neuronal
    lr_model, history, nn_model = create_train_lr_and_nn_model(X_train_scaled, y_train)
    
    # llamamos a funcion de guardado de modelos
    save_trained_models(lr_model, dt_model, rf_model, xgb_model, nn_model)

    # evaluamos modelos (desde evaluator.py) y lo guardamos en un diccionario conjunto
    results = {}
    results["logistic_regression"] = evaluate_sklearn_model(lr_model, X_test_scaled, y_test)
    results["decision_tree"] = evaluate_sklearn_model(dt_model, X_test_preprocessed, y_test)
    results["random_forest"] = evaluate_sklearn_model(rf_model, X_test_preprocessed, y_test)
    results["xgboost"] = evaluate_sklearn_model(xgb_model, X_test_preprocessed, y_test)
    results["neural_network"] = evaluate_nn_model(nn_model, X_test_scaled, y_test, history) # threshold > 0.4

    # creamos tabla comparativa con las métricas de todos los modelos
    comparison_table = create_model_comparison_table(results)
    print("\nTabla comparativa de modelos:")
    print(comparison_table)

    # sacamos predicciones de conjunto test de cada modelo
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_dt = dt_model.predict(X_test_preprocessed)
    y_pred_rf = rf_model.predict(X_test_preprocessed)
    y_pred_xgb = xgb_model.predict(X_test_preprocessed)

    # caso de red neuronal sacamos probabilidades y calculamos predicciones con threshold
    y_score_nn = nn_model.predict(X_test_scaled,verbose=0).reshape(-1)
    y_pred_nn = (y_score_nn > NN_THRESHOLD).astype(int)

    # matrices de confusion
    plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
    plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree")
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
    plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")
    plot_confusion_matrix(y_test, y_pred_nn, "Neural Network")


    # sacamos probabilidades de conjunto de test de cada modelo
    y_score_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_score_dt = dt_model.predict_proba(X_test_preprocessed)[:, 1]
    y_score_rf = rf_model.predict_proba(X_test_preprocessed)[:, 1]
    y_score_xgb = xgb_model.predict_proba(X_test_preprocessed)[:, 1]

    # ROC individuales
    plot_roc_curve(y_test, y_score_lr, "Logistic Regression")
    plot_roc_curve(y_test, y_score_dt, "Decision Tree")
    plot_roc_curve(y_test, y_score_rf, "Random Forest")
    plot_roc_curve(y_test, y_score_xgb, "XGBoost")
    plot_roc_curve(y_test, y_score_nn, "Neural Network")

    # ROC comparativa
    scores_by_model = {
        "logistic_regression": y_score_lr,
        "decision_tree": y_score_dt,
        "random_forest": y_score_rf,
        "xgboost": y_score_xgb,
        "neural_network": y_score_nn
    }

    plot_comparative_roc_curve(y_test, scores_by_model)


    ### FEATURE IMPORTANCE 
    rf_feature_importance = create_feature_importance_table(
        rf_model,
        feature_names
    )
    
    xgb_feature_importance = create_feature_importance_table(
        xgb_model,
        feature_names
    )
    
    plot_feature_importance(
        rf_model,
        feature_names,
        model_name="Random Forest",
        top_n=20
    )
    
    plot_feature_importance(
        xgb_model,
        feature_names,
        model_name="XGBoost",
        top_n=20
    )

    # diccionario nombre_modelo: modelos
    trained_models = {
        "logistic_regression": lr_model,
        "decision_tree": dt_model,
        "random_forest": rf_model,
        "xgboost": xgb_model,
        "neural_network": nn_model 
    }

    # modelos requeridos para hacer la comparación
    required_models = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "neural_network"
    ]

    # llamamos a funcion de comapracion
    best_model_name = select_best_model(
        results,
        primary_metric="f1",
        required_models = required_models
    )

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

    elif best_model_name == "logistic_regression":
        # caso de modelo clasico
        joblib.dump(best_model, BEST_MODEL_DIR / "best_model.pkl")

        # info mejor modelo
        metadata = {
            "best_model_name": best_model_name,
            "best_model_type": "logistic_regression"
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