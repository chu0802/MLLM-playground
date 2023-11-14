from src.datasets.vqa import BaseMultiAnswersVQADataset
import json


class VizWizVQADataset(BaseMultiAnswersVQADataset):
    def load_data(self, is_train=False):
        annotations = json.load(open(f"{self.data_root}/{self.split}_vqa.json", "r"))
        for ann in annotations:
            answer_dict = (
                self.parse_answer(ann["answers"])
                if is_train
                else {"answers": ann["answers"]}
            )
            image_path = f"{self.data_root}/images/{ann['image']}"
            self.image_path_list.append(image_path)
            self.answer_weight_list.append(answer_dict)
            self.question_list.append(ann["question"])


class VizWizVQATrainDataset(VizWizVQADataset):
    def load_data(self):
        super().load_data(is_train=True)


class VizWizVQAEvalDataset(VizWizVQADataset):
    def load_data(self):
        super().load_data(is_train=False)
