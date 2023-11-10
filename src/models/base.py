import torch.nn as nn
import torch
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = None

    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @abstractmethod
    def parse_images(self, images):
        raise NotImplementedError

    def parse_text_input(self, questions, prompt=None):
        if not isinstance(questions, list):
            questions = [questions]
        if not prompt:
            return questions
        return [prompt.format(question) for question in questions]
