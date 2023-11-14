from dataclasses import dataclass
from typing import Union


@dataclass
class AccMeter:
    num_correct: Union[int, float] = 0
    num_total: int = 0

    def _check_numeric(self, other):
        assert type(other) in [int, float, bool]

    def __add__(self, other):
        if not isinstance(other, AccMeter):
            self._check_numeric(other)
            return AccMeter(self.num_correct + other, self.num_total + 1)
        return AccMeter(
            self.num_correct + other.num_correct, self.num_total + other.num_total
        )

    def __sub__(self, other):
        if not isinstance(other, AccMeter):
            self._check_numeric(other)
            return AccMeter(self.num_correct - other, self.num_total - 1)
        return AccMeter(
            self.num_correct - other.num_correct, self.num_total - other.num_total
        )

    def get_acc(self):
        return self.num_correct / self.num_total


if __name__ == "__main__":
    a = AccMeter(1, 2)
    b = AccMeter(2, 3)
    c = a + b
    print(c)
    d = a - b
    print(d)
