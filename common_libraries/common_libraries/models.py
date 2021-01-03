import pickle
from typing import Any

import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_rmse(lr_model: Any, X: pd.DataFrame, Y: pd.DataFrame) -> float:
    """
    Calculates RMSE of calculated preditions
    """
    Y_pred = lr_model.predict(X)
    return mean_squared_error(Y, Y_pred, squared=False)


def save(model: Any, model_name: str):
    with open(model_name, "wb") as mw:
        pickle.dump(model, mw)


def load_model(model_path: str):
    with open(model_path, "rb") as mr:
        return pickle.load(mr)
