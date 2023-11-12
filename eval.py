import os

from src.utils.config import get_config
from src.tasks import get_task
import logging
import sys
from src.datasets import get_dataset
from src.models import get_model
from src.evaluater import get_evaluater
from src.datasets.utils import get_dataloaders
from src.utils.seed import setup_seeds
from src.trainer.trainer import Trainer

log_format = """[%(levelname)s] [%(asctime)s] %(message)s"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)


def main(config):
    setup_seeds(config.task.seed)

    logging.info("Start Inference")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.task.device)

    task = get_task(config)
    dataloaders = get_dataloaders(config)
    evaluater = get_evaluater(config.dataset.name)
    model = get_model(config)

    trainer = Trainer(task, model, dataloaders, config)

    score = trainer.evaluate(evaluater)

    logging.info(
        f"Model: {config.model.name} | Dataset: {config.dataset.name} | Result: {score}"
    )


if __name__ == "__main__":
    main(get_config())
