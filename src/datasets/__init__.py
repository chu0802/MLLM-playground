from src.datasets.vqa.vizwiz import VizWizVQADataset
from src.datasets.vqa.science_qa import ScienceQADataset
from src.datasets.vqa.text_vqa import TextVQADataset


DATASET_DICT = {
    "VizWiz": VizWizVQADataset,
    "ScienceQA": ScienceQADataset,
    "TextVQA": TextVQADataset,
}


def get_dataset(dataset_name):
    return DATASET_DICT[dataset_name]
