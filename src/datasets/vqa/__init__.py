from torch.utils.data import Dataset
from abc import abstractmethod
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


def get_image(image):
    if type(image) in [str, Path]:
        return Image.open(image).convert("RGB")
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of image: {type(image)}")


class BaseVQADataset(Dataset):
    def __init__(self, config):
        self.image_path_list = []
        self.question_list = []
        self.answer_list = []
        self.data_root = (
            Path(config.dataset.root) / "VQA_Datasets" / config.dataset.name
        )
        self.split = config.dataset.split
        self.load_data()

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_path_list[idx]
        return {
            "image": get_image(img_path),
            "image_path": img_path,
            "question": question,
            "answers": answers,
        }
