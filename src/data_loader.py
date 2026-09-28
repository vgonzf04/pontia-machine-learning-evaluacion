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
# rutas para guardar datasets preprocesados
TREE_PROCESSED_PATH = Path("data/processed/tree_preprocessed.csv")
SCALED_PROCESSED_PATH = Path("data/processed/scaled_preprocessed.csv")
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

    # comprobamos si existen los archivos de datos preprocesados
    if TREE_PROCESSED_PATH.exists() and SCALED_PROCESSED_PATH.exists():
        # leemos los datos preprocesados para modelos de arbol
        tree_df = pd.read_csv(TREE_PROCESSED_PATH)

        # leemos los datos preprocesados y escalados para Logistic Regression y NN
        scaled_df = pd.read_csv(SCALED_PROCESSED_PATH)

        # comprobamos que tenga datos
        if not tree_df.empty and not scaled_df.empty:
            # CASO DE DATOS PREPREPROCESADOS YA GUARDADOS
            # separamos conjunto de entrenamiento
            tree_train_df = tree_df[tree_df["split"] == "train"]

            # separamos conjunto de test
            tree_test_df = tree_df[tree_df["split"] == "test"]

            # obtenemos variables independientes de entrenamiento
            X_train_preprocessed = tree_train_df.drop(columns=["is_canceled", "split"])

            # obtenemos variables independientes de test
            X_test_preprocessed = tree_test_df.drop(columns=["is_canceled", "split"])

            # obtenemos target de entrenamiento y test
            y_train = tree_train_df["is_canceled"]
            y_test = tree_test_df["is_canceled"]


            # separamos conjunto de entrenamiento
            scaled_train_df = scaled_df[scaled_df["split"] == "train"]

            # separamos conjunto de test
            scaled_test_df = scaled_df[scaled_df["split"] == "test"]

            # obtenemos variables independientes escaladas de entrenamiento
            X_train_scaled = scaled_train_df.drop(columns=["is_canceled", "split"])

            # obtenemos variables independientes escaladas de test
            X_test_scaled = scaled_test_df.drop(columns=["is_canceled", "split"])

            # cargamos el preprocesador de arboles ya entrenado
            tree_preprocessor = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")

            # recuperamos los nombres de las variables transformadas
            feature_names = tree_preprocessor.get_feature_names_out()

            # devolvemos todos los conjuntos ya procesados
            return (X_train_preprocessed, X_test_preprocessed, X_train_scaled, X_test_scaled, y_train, y_test, feature_names)


    # CASO DE QUE NO HAYA DATOS PREPROCESADOS YA GUARDADOS
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


    # obtenemos los nombres de las columnas generadas por cada preprocesador
    tree_feature_names = tree_preprocessor.get_feature_names_out()
    scaled_feature_names = scaled_preprocessor.get_feature_names_out()

    # guardamos transformadores en disco para mantener sus entrenamientos
    joblib.dump(tree_preprocessor, MODELS_DIR / "tree_preprocessor.pkl")
    joblib.dump(scaled_preprocessor, MODELS_DIR / "scaled_preprocessor.pkl")

    # añadimos target y split al conjunto de entrenamiento
    tree_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_preprocessed, columns=tree_feature_names, index=X_train.index)
    tree_test_df = pd.DataFrame.sparse.from_spmatrix(X_test_preprocessed, columns=tree_feature_names, index=X_test.index)

    # convertimos los datos preprocesados a DataFrame
    scaled_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_scaled, columns=scaled_feature_names, index=X_train.index)
    scaled_test_df = pd.DataFrame.sparse.from_spmatrix(X_test_scaled, columns=scaled_feature_names, index=X_test.index)
    
    # añadimos target y split al conjunto de modelos de arbol
    tree_train_df["is_canceled"] = y_train
    tree_train_df["split"] = "train"
    tree_test_df["is_canceled"] = y_test
    tree_test_df["split"] = "test"

    # añadimos target y split al conjunto de los modelos de regresion y red neuronal
    scaled_train_df["is_canceled"] = y_train
    scaled_train_df["split"] = "train"
    scaled_test_df["is_canceled"] = y_test
    scaled_test_df["split"] = "test"
    
    # juntamos train y test preprocesados para modelos de arbol en un unico DataFrame
    tree_df = pd.concat([tree_train_df, tree_test_df])
    # juntamos train y test preprocesados y escalados en un unico DataFrame
    scaled_df = pd.concat([scaled_train_df, scaled_test_df])
    
    # comprobamos que exista directorio y si no lo creamos
    TREE_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # guardamos los datos preprocesados en CSVs distintos
    # index=False evita guardar el indice del DataFrame como una columna adicional
    tree_df.to_csv(TREE_PROCESSED_PATH, index=False)
    scaled_df.to_csv(SCALED_PROCESSED_PATH, index=False)

    # devolvemos conjunto de entrenamiento y test
    return X_train_preprocessed, X_test_preprocessed, X_train_scaled, X_test_scaled, y_train, y_test, tree_feature_names
