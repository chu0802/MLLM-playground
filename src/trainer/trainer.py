from tqdm import tqdm
import json
from pathlib import Path
import datetime
from src.trainer.optimizer import get_optimizer
from src.trainer.lr_scheduler import get_lr_scheduler
from src.utils.download import load_checkpoint
from src.utils.config import dump_config
import torch
import logging
import wandb


class Trainer:
    def __init__(self, task, model, dataloaders, config):
        self.task = task
        self.model = model
        self.dataloaders = dataloaders
        self.config = config

        self.is_train = "train" in config.dataset.split

        if self.is_train:
            self.optimizer = get_optimizer(model, config)
            self.lr_scheduler = get_lr_scheduler(self.optimizer, config)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        self.output_dir = (
            Path(self.config.task.output_dir)
            / self.config.model.name
            / self.config.dataset.name
            / timestamp
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        dump_config(self.config, self.output_dir / "config.json")

    @property
    def max_epoch(self):
        return self.config.task.max_epoch

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def train_loader(self):
        return self.dataloaders["train"]

    @property
    def eval_loader(self):
        return self.dataloaders["eval"]

    def save(self, epoch):
        # avoid the name to be prefixed with "model."
        param_grad_dict = {
            k: v.requires_grad for (k, v) in self.model.model.named_parameters()
        }

        state_dict = self.model.model.state_dict()

        for k in list(state_dict.keys()):
            if k in param_grad_dict.keys() and not param_grad_dict[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }

        save_path = self.output_dir / f"checkpoint_{epoch}.pth"

        logging.info(f"Saving checkpoint at epoch {epoch} to {save_path}.")
        torch.save(save_obj, save_path)

    def load(self, path):
        checkpoint = load_checkpoint(path)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {path}")

    def dump_results(self, results, score):
        detailed_results_path = self.output_dir / "detailed_results.json"
        result_path = self.output_dir / "result.json"

        with detailed_results_path.open("w") as f:
            f.write(json.dumps(results, indent=4))

        with result_path.open("w") as f:
            f.write(json.dumps({"result": score}, indent=4))

    def evaluate(self, evaluater):
        self.model.eval()

        results = []
        for batch in tqdm(self.eval_loader, desc="Evaluation"):
            eval_output = self.task.evaluate_step(self.model, batch)
            results += eval_output
        score = self.task._eval_metrics(results, evaluater)

        self.dump_results(results, score)

        return score

    def train(self):
        self.model.train()
        with tqdm(total=self.max_epoch * len(self.train_loader)) as pbar:
            for epoch in range(self.max_epoch):
                for i, batch in enumerate(self.train_loader):
                    loss, loss_dict = self.task.train_step(self.model, batch)

                    self.lr_scheduler.step(cur_epoch=epoch, cur_step=i)

                    if i % 50 == 0:
                        wandb.log(
                            {
                                "lr": self.lr,
                                "loss": loss.item(),
                            }
                        )

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    pbar.update(1)

                self.save(epoch)
