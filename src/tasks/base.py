from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod
from pathlib import Path
import datetime
import json


class BaseTask:
    def __init__(self, config):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = config
        self.output_dir = (
            Path(self.config.task.output_dir)
            / self.config.model.name
            / self.config.dataset.name
            / timestamp
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            collate_fn=lambda batch: {
                key: [dict[key] for dict in batch] for key in batch[0]
            },
        )

    @abstractmethod
    def evaluate_step(self, model, batch):
        raise NotImplementedError

    @abstractmethod
    def _eval_metrics(self, results, eval_cls):
        raise NotImplementedError

    def _dump_results(self, results, score):
        detailed_results_path = self.output_dir / "detailed_results.json"
        result_path = self.output_dir / "result.json"

        with detailed_results_path.open("w") as f:
            f.write(json.dumps(results, indent=4))

        with result_path.open("w") as f:
            f.write(json.dumps({"result": score}, indent=4))

    def evaluate(self, model, dataloader, eval_cls=None):
        model.eval()

        results = []
        for batch in tqdm(dataloader, desc="Evaluation"):
            eval_output = self.evaluate_step(model, batch)
            results += eval_output
        score = self._eval_metrics(results, eval_cls)

        self._dump_results(results, score)

        return score
