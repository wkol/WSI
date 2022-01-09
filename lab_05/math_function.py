from typing import Callable, Any, Optional
import numpy as np


class MissingDerivative(Exception):
    pass


class MathFunction:
    def __init__(self, fun: Callable, d_fun: Optional[Callable] = None, *args: Any) -> None:
        self.args = args
        self.fun = fun
        self.derivative_function = d_fun

    def derivative(self, x: np.ndarray, *args: Any) -> np.ndarray:
        if self.derivative_function is None:
            raise MissingDerivative("Derivative function is not provided")
        return self.derivative_function(x, *args)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fun(x, *self.args)


def neural_function(x: np.ndarray) -> np.ndarray:
    return x * x * np.sin(x) + 10 * 10 * np.sin(x) * np.cos(x)
