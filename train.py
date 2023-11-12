import os

from src.utils.config import get_config
from src.tasks import get_task
import logging
import sys
from src.models import get_model
from src.evaluater import get_evaluater
from src.datasets.utils import get_dataloaders
from src.utils.seed import setup_seeds
from src.trainer.trainer import Trainer
from src.utils.wandb import wandb_logger

log_format = """[%(levelname)s] [%(asctime)s] %(message)s"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)


@wandb_logger
def main(config):
    setup_seeds(config.task.seed)

    logging.info("Start Inference")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.task.device)

    task = get_task(config)
    model = get_model(config)
    dataloaders = get_dataloaders(config)
    trainer = Trainer(task, model, dataloaders, config)

    trainer.train()


if __name__ == "__main__":
    main(get_config(train=True))
