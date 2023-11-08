from src.datasets.vqa import BaseVQADataset
import json


class VizWizVQADataset(BaseVQADataset):
    def __init__(self, config):
        super().__init__(config)

    def load_data(self, split="val"):
        annotations = json.load(open(f"{self.data_root}/{split}_vqa.json", "r"))
        for ann in annotations:
            image_path = f"{self.data_root}/{ann['image']}"
            self.image_path_list.append(image_path)
            self.answer_list.append(ann["answers"])
            self.question_list.append(ann["question"])
