from src.datasets.vqa.vizwiz import VizWizVQATrainDataset, VizWizVQAEvalDataset
from src.datasets.vqa.science_qa import ScienceQATrainDataset, ScienceQAEvalDataset
from src.datasets.vqa.text_vqa import TextVQADataset
from src.datasets.vqa.ok_vqa import OKVQATrainDataset, OKVQAEvalDataset

DATASET_DICT = {
    "VizWiz": {
        "train": VizWizVQATrainDataset,
        "eval": VizWizVQAEvalDataset,
    },
    "ScienceQA": {
        "train": ScienceQATrainDataset,
        "eval": ScienceQAEvalDataset,
    },
    "TextVQA": {
        "train": TextVQADataset,
        "eval": TextVQADataset,
    },
    "OKVQA": {
        "train": OKVQATrainDataset,
        "eval": OKVQAEvalDataset,
    },
}


def get_dataset(config, split_type):
    split_config = config.dataset.split.get(split_type, None)
    return DATASET_DICT[config.dataset.name][split_type](config, split_config["name"])
