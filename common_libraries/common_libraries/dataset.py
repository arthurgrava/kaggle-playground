from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error


def random_split_on_data(data: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data passed into two chunks using `train_size` as proportion
    """
    split = np.random.rand(len(data)) < train_size
    train = data[split]
    test = data[~split]
    return train, test


def split_dataset_supervised(data: pd.DataFrame, train_cols: List[str], y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into X (to calculate the output) and Y (to calculate the error)
    """
    temp = data.copy(deep=True)
    X = temp[train_cols]
    Y = temp[[y_col]]
    return X, Y


def calculate_rmse(lr_model: Any, X: pd.DataFrame, Y: pd.DataFrame) -> float:
    """
    Calculates RMSE of calculated preditions
    """
    Y_pred = lr_model.predict(X)
    return mean_squared_error(Y, Y_pred, squared=False)
