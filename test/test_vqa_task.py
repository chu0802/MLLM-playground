from src.utils import get_config
from src.tasks.vqa import VQATask
from test.test_model import DummyModel
from test.test_dataset import DummyVQADataset
from test.test_eval import DummyVQAEval


def test_vqa_task():
    config = get_config()
    vqa_task = VQATask(config)

    model = DummyModel()
    dummy_dataset = DummyVQADataset(size=100)
    dataloader = vqa_task.build_dataloader(dummy_dataset)

    vqa_task.evaluate(model, dataloader, eval_cls=DummyVQAEval)


if __name__ == "__main__":
    test_vqa_task()
