import torch
import torchvision.transforms as transforms

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class DummyModel:
    def __init__(self):
        pass

    def eval(self):
        pass

    def batch_generate(self, images, questions, max_len, min_len, num_beams):
        images = (
            torch.cat([TRANSFORM(image) for image in images], dim=0).bfloat16().cuda()
        )
        return range(len(images))
