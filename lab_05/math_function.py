from typing import Callable, Any
from math import cos, sin


class MathFunction:
    def __init__(self, fun: Callable, *args: Any) -> None:
        self.args = args
        self.fun = fun

    def __call__(self, x: float) -> float:
        return self.fun(x, *self.args)


def neural_function(x: float) -> float:
    return x * x * sin(x) + 10 * 10 * sin(x) * cos(x)
