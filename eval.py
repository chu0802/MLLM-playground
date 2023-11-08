import os

from src.utils import Config, parse_args
from src.tasks import TASK_DICT
from test.test_model import DummyModel
from test.test_eval import DummyVQAEval
import logging
import sys
from src.datasets import DATASET_DICT
from src.models import MODEL_DICT
from src.datasets.utils import sample_dataset

log_format = """[%(levelname)s] [%(asctime)s] %(message)s"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)


def main(config):
    logging.info("Start Inference")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.task.device)

    task = TASK_DICT[config.task.name](config)

    model = MODEL_DICT[config.model.name](config)
    dataset = sample_dataset(
        DATASET_DICT[config.dataset.name](config),
        config.dataset.sample_num,
        config.dataset.sample_seed,
    )
    dataloader = task.build_dataloader(dataset)

    score = task.evaluate(model, dataloader, eval_cls=DummyVQAEval)

    logging.info(
        f"Model: {config.model.name} | Dataset: {config.dataset.name} | Result: {score}"
    )


if __name__ == "__main__":
    main(Config(parse_args()).config)
