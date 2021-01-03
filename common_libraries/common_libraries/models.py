import pickle
from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(lr_model: Any, X: pd.DataFrame, Y: pd.DataFrame) -> float:
    """
    Calculates RMSE of calculated preditions
    """
    Y_pred = lr_model.predict(X)
    return mean_squared_error(Y, Y_pred, squared=False)


def calculate_rmse_log(lr_model: Any, X: pd.DataFrame, Y: pd.DataFrame) -> float:
    """
    Calculates RMSE of the predictions applying log in the difference
    """
    Y_ini = pd.DataFrame(list(Y[Y.columns[0]]), columns=["Real"])
    Y_pred = pd.DataFrame(lr_model.predict(X), columns=["Pred"])

    df = pd.concat([Y_ini, Y_pred], axis=1)
    df["log"] = df.apply(
        lambda r: np.log( (r["Real"] - r["Pred"]) ** 2)
        , axis=1
    )

    return (df["log"].sum() / df.shape[0]) ** (.5)


def save(model: Any, model_name: str):
    with open(model_name, "wb") as mw:
        pickle.dump(model, mw)


def load_model(model_path: str):
    with open(model_path, "rb") as mr:
        return pickle.load(mr)
