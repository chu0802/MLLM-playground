from src.evaluater.parse_tokens import has_word, equivalent
from src.evaluater.base import BaseEvaluater
from src.utils.metrics import AccMeter


class BaseVQAEvaluater(BaseEvaluater):
    def __init__(self, config):
        self.config = config
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

    def get_eval_metrics(self, results):
        score = AccMeter()
        for res in results:
            eval_res = self.evaluate(res["pred_answer"], res["answers"])
            res["correct"] = eval_res
            score += eval_res

        return score


class VQAEvaluater(BaseEvaluater):
    def evaluate(self, answer, gt_answers):
        if type(gt_answers) == list:
            return any([has_word(answer, gt_answer) for gt_answer in gt_answers])
        else:
            return has_word(answer, gt_answers)


class MrrEvaluater(BaseEvaluater):
    def evaluate(self, answer, gt_answers):
        assert type(gt_answers) == list
        for i in range(len(gt_answers)):
            if has_word(answer, gt_answers[i]):
                return 1 / (i + 1)
        return 0.0


class ScienceQAEvaluater(BaseEvaluater):
    def evaluate(self, answer, gt_answers):
        gt_choice, gt_direct_answer = gt_answers[0].split(") ")
        pred_direct_answer = answer.split(") ")[-1]
        return (
            equivalent(answer, gt_choice)
            or has_word(answer, gt_direct_answer)
            or has_word(gt_direct_answer, pred_direct_answer)
        )


class VQAV2Evaluater(BaseEvaluater):
    def evaluate(self, answer, gt_answers):
        assert type(gt_answers) == list
        return min(
            sum([has_word(answer, gt_answer) for gt_answer in gt_answers]) / 3, 1
        )
