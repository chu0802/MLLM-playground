import random


class DummyVQAEval:
    def __init__(self):
        pass

    def evaluate(self, a, b):
        return bool(random.randint(0, 1))
