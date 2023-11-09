import os

from config import get_config
from src.tasks import get_task
import logging
import sys
from src.datasets import get_dataset
from src.models import get_model
from src.evaluater import get_evaluater
from src.datasets.utils import sample_dataset

log_format = """[%(levelname)s] [%(asctime)s] %(message)s"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)


def main(config):
    logging.info("Start Inference")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.task.device)

    task = get_task(config.task.name)(config)
    model = get_model(config.model.name)(config)
    eval_cls = get_evaluater(config.dataset.name)

    dataset = sample_dataset(
        get_dataset(config.dataset.name)(config),
        config.dataset.sample_num,
        config.dataset.sample_seed,
    )
    dataloader = task.build_dataloader(dataset)

    score = task.evaluate(model, dataloader, eval_cls=eval_cls)

    logging.info(
        f"Model: {config.model.name} | Dataset: {config.dataset.name} | Result: {score}"
    )


if __name__ == "__main__":
    main(get_config())
