from src.models.base import BaseModel
from lavis.models.blip2_models.blip2_t5_instruct import Blip2T5Instruct
from lavis.models import load_preprocess
import torch
from omegaconf import OmegaConf


class Blip2(BaseModel):
    def __init__(self, config):
        super().__init__()
        model_default_config = OmegaConf.load(config.model.config_path)
        self.model = Blip2T5Instruct(**dict(model_default_config.model))

        self.vis_processors, self.text_processors = load_preprocess(
            model_default_config.preprocess
        )
        self.model.load_from_pretrained(config.model.model_path)

    def train(self):
        self.model.Qformer.train()
        self.model.t5_proj.train()

    def eval(self):
        self.model.eval()

    def get_params(self, weight_decay, lr_scale=1):
        weight_decay_params, non_weight_decay_params = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "ln" in name or "bn" in name:
                non_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

        optim_params = [
            {
                "params": weight_decay_params,
                "weight_decay": weight_decay,
                "lr_scale": lr_scale,
            },
            {
                "params": non_weight_decay_params,
                "weight_decay": 0,
                "lr_scale": lr_scale,
            },
        ]

        return optim_params

    def parse_images(self, images):
        if not isinstance(images, list):
            images = [images]
        images = torch.stack(
            [self.vis_processors["eval"](x) for x in images], dim=0
        ).to(self.device)
        return images

    def parse_questions(self, questions, mode="train"):
        return [self.text_processors[mode](question) for question in questions]

    # TODO: move this procedure to dataset
    def forward(self, batch):
        images = self.parse_images(batch["image"])
        questions = self.parse_questions(batch["question"])
        samples = {
            "image": images,
            "text_input": questions,
            "text_output": batch["answers"],
        }

        if "n_answers" in batch:
            samples["n_answers"] = batch["n_answers"]
        return self.model(samples=samples)

    @torch.no_grad()
    def generate(
        self,
        images,
        questions,
        max_len=30,
        min_len=1,
        num_beams=5,
        length_penalty=-1,
        prompt=None,
    ):
        images = self.parse_images(images)
        input_prompts = self.parse_text_input(questions, prompt)

        answer = self.model.generate(
            samples={"image": images, "prompt": input_prompts},
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        return answer


if __name__ == "__main__":
    from src.utils.config import get_config

    blip2 = Blip2(get_config())
    blip2 = blip2.cuda()
