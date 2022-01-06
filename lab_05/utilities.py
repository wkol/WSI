from typing import Tuple
from math_function import MathFunction
import numpy as np

def prepare_dataset(fun: MathFunction, start: float, end: float, samples_n: int) -> Tuple[np.ndarray]:
    input_features = np.linspace(start, end, samples_n)
    vfunc = np.vectorize(fun)
    return input_features, vfunc(input_features)


print(prepare_dataset(MathFunction(lambda x: x * x), -10, 10, 20))