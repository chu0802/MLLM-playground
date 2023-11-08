from src.datasets.vqa.vizwiz import VizWizVQADataset


DATASET_DICT = {"VizWiz": VizWizVQADataset}


def get_dataset(dataset_name):
    return DATASET_DICT[dataset_name]
