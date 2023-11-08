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


class VQATask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def evaluate_step(self, model, batch):
        outputs = model.generate(
            batch["image"],
            batch["question"],
            max_len=self.config.task.max_len,
            min_len=self.config.task.min_len,
            num_beams=self.config.task.num_beams,
        )

        return [
            {
                "question": question,
                "pred_answer": output,
                "answers": answer,
                "image_path": image_path,
            }
            for question, output, answer, image_path in zip(
                batch["question"], outputs, batch["answers"], batch["image_path"]
            )
        ]

    def _eval_metrics(self, results, eval_cls):
        evaluater = eval_cls()

        correct = 0
        for res in results:
            eval_res = evaluater.evaluate(res["pred_answer"], res["answers"])
            res["correct"] = eval_res
            correct += eval_res

        score = correct / len(results)
        return score
