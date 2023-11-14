from src.datasets.vqa import BaseMultiAnswersVQADataset
import json


class TextVQADataset(BaseMultiAnswersVQADataset):
    def load_data(self):
        data = json.load(
            open(f"{self.data_root}/TextVQA_0.5.1_{self.split}.json", "r")
        )["data"]
        image_dir = self.data_root / (
            "test_images" if self.split == "test" else "train_images"
        )

        for d in data:
            self.image_path_list.append((image_dir / f"{d['image_id']}.jpg").as_posix())
            self.question_list.append(d["question"])
            self.answer_weight_list.append({"answers": d["answers"]})


if __name__ == "__main__":
    from src.utils.config import get_config

    config = get_config()
    config.dataset.name = "TextVQA"
    science_qa = TextVQADataset(config, split=config.dataset.split.train.name)

    l = science_qa[0]
    pass
