from src.evaluater.vqa import VQAEvaluater, VQAV2Evaluater, ScienceQAEvaluater

__all__ = [
    "VQAEvaluater",
    "VQAV2Evaluater",
    "ScienceQAEvaluater",
]

EVALUATER_DICT = {
    "ScienceQA": ScienceQAEvaluater,
    "VizWiz": VQAV2Evaluater,
}


def get_evaluater(dataset_name):
    return EVALUATER_DICT.get(dataset_name, VQAEvaluater)()
