from src.models.blip2.blip2 import Blip2

__all__ = [
    "Blip2",
]

MODEL_DICT = {"Blip2": Blip2}


def get_model(config):
    model = MODEL_DICT[config.model.name](config)
    model.cuda()
    return model
