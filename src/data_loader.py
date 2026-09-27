import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

# ruta de dataset para predecir
DATA_PREDICT_PATH = Path("data/raw/data_predict.csv")
# ruta de csv para entrenamiento y test
DATA_RAW_PATH = Path("data/raw/dataset_practica_final.csv")
# ruta para guaradar dataset preprocesado
DATA_PROCESSED_PATH = Path("data/processed/dataset_preprocessed.csv")
# ruta de modelos y preprocesador
MODELS_DIR = Path("models/tests")


def preprocessed_for_tree_prediction():
    # cargar datos para predictor
    df = pd.read_csv(DATA_PREDICT_PATH)

    # eliminamos columnas que no aportan nada
    # company en la mayoria de sus casos tiene valores nulos y ademas es un id de la compañia lo cual no aporta info relevante
    # reservation_status y reservation_status_code puede contener la info que tienen que predecir los modelos 
    X = df.drop(columns=["company", "reservation_status", "reservation_status_date"])

    # cargamos el transformer
    tree_preprocessor = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")

    # SOLO transformamos X con transformer entrenado
    X_preprocessed = tree_preprocessor.transform(X)

    # devolvemos X preprocesado para la prediccion con modelos scikit-learn
    return X_preprocessed

def preprocessed_and_scaled_for_prediction():
    # cargar datos para predictor
    df = pd.read_csv(DATA_PREDICT_PATH)

    # eliminamos columnas que no aportan nada
    # company en la mayoria de sus casos tiene valores nulos y ademas es un id de la compañia lo cual no aporta info relevante
    # reservation_status y reservation_status_code puede contener la info que tienen que predecir los modelos 
    X = df.drop(columns=["company", "reservation_status", "reservation_status_date"])

    # cargamos el transformer
    scaled_preprocessor = joblib.load(MODELS_DIR / "scaled_preprocessor.pkl")

    # SOLO transformamos X con tranformer entrenado
    X_preprocessed_scaled = scaled_preprocessor.transform(X)

    # devolvemos X preprocesado para la prediccion con una red neuronal
    return X_preprocessed_scaled

# preprocesamos dataset
def get_preprocessed_data():

    # comprobamos si existe el archivo de datos preprocesados
    if DATA_PROCESSED_PATH.exists():
        # si existe lo leemos
        df_preprocessed = pd.read_csv(DATA_PROCESSED_PATH)

        # comprobamos que tenga datos
        if not df_preprocessed.empty:
            # caso de tener datos ya preprocesados 
            # separamos conjunto de entrenamiento 
            train_df = df_preprocessed[df_preprocessed["split"] == "train"]
            # separamos conjunto de test
            test_df = df_preprocessed[df_preprocessed["split"] == "test"]

            # separamos en X, y para train
            X_train_preprocessed = train_df.drop(columns=["is_canceled", "split"])
            y_train = train_df["is_canceled"]

            # separamos en X, y para test
            X_test_preprocessed = test_df.drop(columns=["is_canceled", "split"])
            y_test = test_df["is_canceled"]

            # sacamos feature importance desde el transformer
            tree_preprocessor = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")
            feature_names = tree_preprocessor.get_feature_names_out()

            # devolvemos onjunto de entrenamiento y test preprocesados para modelos scikit-learn
            return X_train_preprocessed, X_test_preprocessed, y_train, y_test, feature_names

    # caso de que no haya datos preprocesados guardados
    # leer dataset desde csv
    df = pd.read_csv(DATA_RAW_PATH)

    # eliminamos columnas que no aportan nada
    # company en la mayoria de sus casos tiene valores nulos y ademas es un id de la compañia lo cual no aporta info relevante
    # reservation_status y reservation_status_code puede contener la info que tienen que predecir los modelos 
    df = df.drop(columns=["company", "reservation_status", "reservation_status_date"])

    # guardamos columnas categoricas y numericas por separado
    categorical_columns = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type"]
    numerical_columns = ["lead_time", "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "agent", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests"]
    
    # preprocesador para arboles
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    tree_preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])


    # preprocesador para regresion logistica y red neuronal
    numerical_scaled_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    scaled_preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_scaled_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])

    # division del DataFrame en:
    # variables independientes
    X = df.drop(columns=["is_canceled"])
    # variable dependiente, tarjet
    y = df["is_canceled"]

    # separamos X e y en conjunto de entrenamiento (80%) y conunto de test (20%) semilla random habitual y mantenemos proporcion de clases con stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

    # entrenamos transformer y transformamos conjunto X de entrenamiento para arboles
    X_train_preprocessed = tree_preprocessor.fit_transform(X_train)
    # SOLO transformamos el conjunto de test, pero no entrenamos el transformer para arboles
    X_test_preprocessed = tree_preprocessor.transform(X_test)

    # entrenamos transformer y transformamos conjunto X de entrenamiento para lr y nn
    X_train_scaled = scaled_preprocessor.fit_transform(X_train)
    # SOLO transformamos el conjunto de test, pero no entrenamos el transformer para lr y nn
    X_test_scaled = scaled_preprocessor.transform(X_test)


    # guardamos nombres
    feature_names = tree_preprocessor.get_feature_names_out()

    # guardamos transformadores en disco para mantener sus entrenamientos
    joblib.dump(tree_preprocessor, MODELS_DIR / "tree_preprocessor.pkl")
    joblib.dump(scaled_preprocessor, MODELS_DIR / "scaled_preprocessor.pkl")

    # añadimos target y split al conjunto de entrenamiento
    train_df = X_train_preprocessed.copy()
    train_df["is_canceled"] = y_train
    train_df["split"] = "train"

    # añadimos target y split al conjunto de test
    test_df = X_test_preprocessed.copy()
    test_df["is_canceled"] = y_test
    test_df["split"] = "test"

    # juntamos ambos datasets
    sklearn_df = pd.concat([train_df, test_df])

    # comprobamos que existe el directorio 
    DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # guardamos csv sin indices
    sklearn_df.to_csv(DATA_PROCESSED_PATH, index=False)

    # devolvemos conjunto de entrenamiento y test
    return X_train_preprocessed, X_test_preprocessed, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
