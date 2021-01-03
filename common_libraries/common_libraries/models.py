import pickle
from typing import Any


def save(model: Any, model_name: str):
    with open(model_name, "wb") as mw:
        pickle.dump(model, mw)


def load_model(model_path: str):
    with open(model_path, "rb") as mr:
        return pickle.load(mr)
