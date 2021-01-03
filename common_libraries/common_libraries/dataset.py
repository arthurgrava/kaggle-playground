from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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


def fit_categorical_encoder(data: pd.DataFrame, column: str) -> OneHotEncoder:
    """
    Given a DF with columns representing categories, we transform it in arrays, e.g., a column type
    with values A and B will become two columns A and B.
    """
    encoder = OneHotEncoder().fit(data[[column]])
    return encoder


def transform_with_fitted_encoder(data: pd.DataFrame, column: str, encoder: OneHotEncoder) -> pd.DataFrame:
    temporary = pd.DataFrame(
        encoder.transform(data[[column]]).toarray(),
        columns=list(encoder.categories_[0]),
    )
    ctn = pd.concat([data, temporary], axis=1)
    del ctn[column]
    return ctn
