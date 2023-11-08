from src.evaluater.parse_tokens import has_word, equivalent


class VQAEval:
    def __init__(self):
        pass

    def evaluate(self, answer, gt_answers):
        if type(gt_answers) == list:
            return any([has_word(answer, gt_answer) for gt_answer in gt_answers])
        else:
            return has_word(answer, gt_answers)

    def evaluate_MRR(self, answer, gt_answers):
        assert type(gt_answers) == list
        for i in range(len(gt_answers)):
            if has_word(answer, gt_answers[i]):
                return 1 / (i + 1)
        return 0.0


class ScienceQAEval(VQAEval):
    def evaluate(self, answer, gt_answers):
        gt_choice, gt_direct_answer = gt_answers.split(") ")
        return equivalent(answer, gt_choice) or has_word(answer, gt_direct_answer)
