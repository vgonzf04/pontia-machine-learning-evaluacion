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


def preprocessed_for_sklearn_prediction():
    # cargar datos para predictor
    df = pd.read_csv(DATA_PREDICT_PATH)

    # eliminamos columnas que no aportan nada
    # company en la mayoria de sus casos tiene valores nulos y ademas es un id de la compañia lo cual no aporta info relevante
    # reservation_status y reservation_status_code puede contener la info que tienen que predecir los modelos 
    X_preprocessed = df.drop(columns=["company", "reservation_status", "reservation_status_date"])

    # devolvemos X preprocesado para la prediccion con modelos scikit-learn
    return X_preprocessed

def preprocessed_for_nn_prediction():
    # cargar datos para predictor
    df = pd.read_csv(DATA_PREDICT_PATH)

    # eliminamos columnas que no aportan nada
    # company en la mayoria de sus casos tiene valores nulos y ademas es un id de la compañia lo cual no aporta info relevante
    # reservation_status y reservation_status_code puede contener la info que tienen que predecir los modelos 
    X = df.drop(columns=["company", "reservation_status", "reservation_status_date"])

    # cargamos el preprocesador
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_nn.pkl")

    # SOLO transformamos X con el preprocesador ya entrenado, recibido como parametro
    X_preprocessed = preprocessor.transform(X)

    # devolvemos X preprocesado para la prediccion con una red neuronal
    return X_preprocessed

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

            # devolvemos onjunto de entrenamiento y test preprocesados para modelos scikit-learn
            return X_train_preprocessed, X_test_preprocessed, y_train, y_test

    # caso de que no haya datos preprocesados guardados
    # leer dataset desde csv
    df = pd.read_csv(DATA_RAW_PATH)

    # eliminamos columnas que no aportan nada
    # company en la mayoria de sus casos tiene valores nulos y ademas es un id de la compañia lo cual no aporta info relevante
    # reservation_status y reservation_status_code puede contener la info que tienen que predecir los modelos 
    df = df.drop(columns=["company", "reservation_status", "reservation_status_date"])

    # division del DataFrame en:
    # variables independientes
    X = df.drop(columns=["is_canceled"])
    # variable dependiente, tarjet
    y = df["is_canceled"]

    # separamos X e y en conjunto de entrenamiento (80%) y conunto de test (20%) semilla random habitual y mantenemos proporcion de clases con stratify
    X_train_preprocessed, X_test_preprocessed, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

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
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

# preprocesamos dataset para train y test de red neuronal
def get_preprocessed_data_for_nn():

    # obtenemos datos preprocesados
    X_train_preprocessed, X_test_preprocessed, y_train, y_test = get_preprocessed_data()

    # guardamos columnas categoricas y numericas por separado
    categorical_columns = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type"]
    numerical_columns = ["lead_time", "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "agent", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests"]
    
    # creamos transformador de columnas numericas
    numerical_transformer = Pipeline(steps=[
        # imputer para sustituir los valores nulos por la mediana
        ("imputer", SimpleImputer(strategy="median")),
        # scaler para tener misma escala en todos los valores, media =0 y desviacion tipica =1
        ("scaler", StandardScaler())
    ])

    # creamos transformador de columnas categoricas
    categorical_transformer = Pipeline(steps=[
        # # imputer para sustituir los valores nulos por el valor mas frecuente
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # encoder para convertir categoricas en numericas
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # preprocesador para aplicar cada transformador en las columnas que le corresponden
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])

    # entrenamos preprocesador y transformamos conjunto X de entrenamiento
    X_train_preprocessed_nn = preprocessor.fit_transform(X_train_preprocessed)
    # SOLO transformamos el conjunto de test, pero no entrenamos el preprocesador
    X_test_preprocessed_nn = preprocessor.transform(X_test_preprocessed)

    # guardamos el preprocesador en disco para mantener su entrenamiento 
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor_nn.pkl")

    # devolvemos preprocesador, conjunto de entrenamiento y test preprocesados para una red neuronal
    return X_train_preprocessed_nn, X_test_preprocessed_nn, y_train, y_test
