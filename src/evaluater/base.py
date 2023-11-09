from abc import abstractmethod


class BaseEvaluater:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, answer, gt_answers):
        raise NotImplementedError
