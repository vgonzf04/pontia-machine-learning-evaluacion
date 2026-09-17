from tensorflow.keras import layers, models

def create_logistic_regression_model():
    return ;
    
def create_decision_tree_model():
    return ;
    
def create_random_forest_model():
    return ;

def create_xgboost_model():
    return ;


def create_neural_network_model(X_train_preprocessed):
    neural_network_model = models.Sequential([
        layers.Input(shape=(X_train_preprocessed.shape[1],), name="o1"),
        layers.Dense(16, activation="relu", name="h1"),
        layers.Dropout(0.1),
        layers.Dense(1, activation="sigmoid", name="o1")
    ])
    neural_network_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return neural_network_model