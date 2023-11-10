from abc import abstractmethod


class BaseTask:
    def __init__(self, config):
        self.config = config
        self.prompt = ""

    def train_step(self, model, batch):
        output = model(batch["image"], batch["question"], self.prompt)
        loss_dict = {}
        for k, v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    @abstractmethod
    def evaluate_step(self, model, batch):
        raise NotImplementedError

    @abstractmethod
    def _eval_metrics(self, results, eval_cls):
        raise NotImplementedError
