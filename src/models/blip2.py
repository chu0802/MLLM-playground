from lavis.models import load_model_and_preprocess
import torch
from src.models import BaseModel


class Blip2(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xl",
            is_eval=config.model.evaluate,
            device=self.device,
        )

    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self, images, questions, max_len=30, min_len=1, num_beams=5, prompt=None
    ):
        if not isinstance(images, list):
            images = [images]
        images = torch.stack(
            [self.vis_processors["eval"](x) for x in images], dim=0
        ).to(self.device)
        input_prompts = self._parse_prompt(questions, prompt)

        answer = self.model.generate(
            samples={"image": images, "prompt": input_prompts},
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
        )

        return answer
