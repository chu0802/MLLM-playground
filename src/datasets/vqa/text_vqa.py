from src.datasets.vqa import BaseVQADataset
import json


class TextVQADataset(BaseVQADataset):
    def load_data(self):
        data = json.load(
            open(f"{self.data_root}/TextVQA_0.5.1_{self.split}.json", "r")
        )["data"]
        image_dir = self.data_root / (
            "test_iamges" if self.split == "test" else "train_images"
        )

        for d in data:
            self.image_path_list.append((image_dir / f"{d['image_id']}.jpg").as_posix())
            self.question_list.append(d["question"])
            self.answer_list.append(d["answers"])
