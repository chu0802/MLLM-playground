from src.tasks.base import BaseTask
from src.utils.metrics import AccMeter


class VQATask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.prompt = config.task.prompt

    def evaluate_step(self, model, batch):
        outputs = model.generate(
            batch["image"],
            batch["question"],
            max_len=self.config.task.max_len,
            min_len=self.config.task.min_len,
            num_beams=self.config.task.num_beams,
            prompt=self.prompt,
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

    def _eval_metrics(self, results, evaluater):
        score = AccMeter()
        for res in results:
            eval_res = evaluater.evaluate(res["pred_answer"], res["answers"])
            res["correct"] = eval_res
            score += eval_res

        return score
