import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _parse_prompt(self, questions, prompt=None):
        if not isinstance(questions, list):
            questions = [questions]
        if not prompt:
            return questions
        return [prompt.format(question) for question in questions]
