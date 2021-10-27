from typing import Callable, Any
import numpy as np


class MathFunction:
    def __init__(self, fun: Callable, *args: Any) -> None:
        self.args = args
        self.fun = fun

    def __call__(self, x) -> float:
        return self.fun(x, *self.args)
        
def fun(x: np.ndarray, a: float, n: int):
    return sum([a ** (i / (n - 1)) * xi * xi for i, xi in enumerate(x)])