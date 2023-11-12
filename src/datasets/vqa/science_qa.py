from src.datasets.vqa import BaseMCQCustomDataset
import datasets
import json
from PIL import Image
import io
from pathlib import Path
from src.datasets.vqa import get_image


class ScienceQADataset(BaseMCQCustomDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        if not (self.data_root / f"{self.split}_anns.json").exists():
            self.download_dataset()
        self.load_data()

    def load_data(self):
        annotations = json.load((self.data_root / f"{self.split}_anns.json").open("r"))
        for ann in annotations:
            self.image_path_list.append((self.data_root / ann["image_path"]).as_posix())
            self.question_list.append(ann["question"])
            self.answer_weight_list.append(self.parse_answer(ann["answer"]))
            self.direct_answer_list.append(ann["direct_answer"])

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
                    "direct_answer": sample["choices"][sample["answer"]],
                }
            )

            image = Image.open(io.BytesIO(sample["image"]["bytes"]))

            abs_img_path = self.data_root / annotations[-1]["image_path"]

            if not abs_img_path.exists():
                abs_img_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(abs_img_path)

        with open(f"{self.data_root}/{self.split}_anns.json", "w") as f:
            f.write(json.dumps(annotations, indent=4))


class ScienceQATrainDataset(ScienceQADataset):
    def __getitem__(self, idx):
        question = self.question_list[idx]
        answer = self.direct_answer_list[idx]
        img_path = self.image_path_list[idx]
        return {
            "image": get_image(img_path),
            "image_path": img_path,
            "question": question,
            # TODO: try to remove the list to make it more intuitive.
            "answers": [answer],
        }


class ScienceQAEvalDataset(ScienceQADataset):
    pass


if __name__ == "__main__":
    from src.utils.config import get_config

    config = get_config()
    config.dataset.name = "ScienceQA"
    science_qa = ScienceQADataset(config, split=config.dataset.split.eval.name)

    l = science_qa[0]
    pass
