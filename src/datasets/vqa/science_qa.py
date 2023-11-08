from src.datasets.vqa import BaseVQADataset
import datasets
import json
from PIL import Image
import io
from pathlib import Path


class ScienceQADataset(BaseVQADataset):
    def __init__(self, config):
        self.image_path_list = []
        self.question_list = []
        self.answer_list = []
        self.data_root = (
            Path(config.dataset.root) / "VQA_Datasets" / config.dataset.name
        )
        self.split = config.dataset.split
        self.options = ["A", "B", "C", "D", "E", "F", "G", "H"]

        ann_path = self.data_root / f"{self.split}_anns.json"
        if not ann_path.exists():
            self.download_dataset()

        self.load_data()

    def load_data(self):
        annotations = json.load((self.data_root / f"{self.split}_anns.json").open("r"))
        for ann in annotations:
            self.image_path_list.append((self.data_root / ann["image_path"]).as_posix())
            self.question_list.append(ann["question"])
            self.answer_list.append(ann["answer"])

    def download_dataset(self):
        data = datasets.load_dataset("derek-thomas/ScienceQA", self.split)
        annotations = []
        for sample in data[self.split]:
            if sample["image"] is None:
                continue

            annotations.append(
                {
                    "image_path": f"{self.split}_imgs/{len(annotations):07d}.png",
                    "question": f"{sample['question']} Options: {' '.join([f'({x}) {y}' for x, y in zip(self.options, sample['choices'])])}",
                    "answer": f"({self.options[sample['answer']]}) {sample['choices'][sample['answer']]}",
                }
            )

            image = Image.open(io.BytesIO(sample["image"]["bytes"]))

            abs_img_path = self.data_root / annotations[-1]["image_path"]

            if not abs_img_path.exists():
                abs_img_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(abs_img_path)

        with open(f"{self.data_root}/{self.split}_anns.json", "w") as f:
            f.write(json.dumps(annotations, indent=4))


if __name__ == "__main__":
    from src.utils import get_config

    config = get_config()
    science_qa = ScienceQADataset(config)
