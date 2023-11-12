from src.datasets.vqa import BaseMultiAnswersVQADataset
import json


class DocVQADataset(BaseMultiAnswersVQADataset):
    def load_data(self):
        ann_path = self.data_root / f"{self.split}_v1.0.json"
        data = json.load(open(ann_path, "r"))["data"]
        for d in data:
            self.image_path_list.append(
                (self.data_root / self.split / d["image"]).as_posix()
            )
            self.question_list.append(d["question"])
            self.answer_weight_list.appned(self.parse_answer(d["answers"])).append(
                d["answers"]
            )
