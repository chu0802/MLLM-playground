from src.evaluater.vqa import VQAEval, ScienceQAEval

__all__ = [
    "VQAEval",
    "ScienceQAEval",
]

EVALUATER_DICT = {
    "ScienceQA": ScienceQAEval,
}


def get_evaluater(dataset_name):
    return EVALUATER_DICT.get(dataset_name, VQAEval)
