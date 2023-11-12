from src.datasets.vqa import BaseMultiAnswersVQADataset
import json


class VizWizVQADataset(BaseMultiAnswersVQADataset):
    def load_data(self):
        annotations = json.load(open(f"{self.data_root}/{self.split}_vqa.json", "r"))
        for ann in annotations:
            image_path = f"{self.data_root}/{ann['image']}"
            self.image_path_list.append(image_path)
            self.answer_weight_list.append(self.parse_answer(ann["answers"]))
            self.question_list.append(ann["question"])
