from src.models.base import BaseModel
from src.models.blip2 import Blip2

__all__ = [
    "BaseModel",
    "Blip2",
]

MODEL_DICT = {"Blip2": Blip2}


def get_model(model_name):
    return MODEL_DICT[model_name]
