from torch.utils.data import Dataset
from abc import abstractmethod
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from collections import defaultdict


def get_image(image):
    if type(image) in [str, Path]:
        return Image.open(image).convert("RGB")
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of image: {type(image)}")


class BaseCustomVQADataset(Dataset):
    def __init__(self, config, split):
        self.image_path_list = []
        self.question_list = []
        self.answer_weight_list = []
        self.data_root = (
            Path(config.dataset.root) / "VQA_Datasets" / config.dataset.name
        )
        self.split = split

    def parse_answer(self, answers):
        answer_weight = defaultdict(float)
        if not isinstance(answers, list):
            answers = [answers]
        for answer in answers:
            answer_weight[answer] += 1 / len(answers)
        return {
            "answers": list(answer_weight.keys()),
            "weights": list(answer_weight.values()),
        }

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answer_weight = self.answer_weight_list[idx]
        img_path = self.image_path_list[idx]
        return {
            "image": get_image(img_path),
            "image_path": img_path,
            "question": question,
            "answers": answer_weight["answers"],
            # "weights": answer_weight["weights"],
        }


class BaseMCQCustomDataset(BaseCustomVQADataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        self.options = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.direct_answer_list = []


class BaseVQADataset(BaseCustomVQADataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        self.load_data()


class BaseMultiAnswersVQADataset(BaseVQADataset):
    def __getitem__(self, idx):
        samples = super().__getitem__(idx)
        samples["n_answers"] = len(samples["answers"])
