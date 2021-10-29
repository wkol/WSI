from typing import Callable, Tuple, List
from matplotlib import pyplot as plt
import numpy as np
from math_function import MathFunction, fun
import timeit
from function_optimization import FunctionOptimization

STEP_SIZES = [0.005, 0.01, 0.1, 1]


def test_step_size(algorithm: Callable, a: float, dim: int):
    function = MathFunction(fun, a, dim)
    starting_points = np.random.uniform(-100, 100, dim)
    y_values = [[], [], [], []]
    if algorithm.__name__ == "newton_backtracking_step":
        result = algorithm(function, starting_points, y_values[0])
        return [y_values[0]], [result[1]]
    iterations = [algorithm(function, starting_points, step, y_values[i])[1] for i, step in enumerate(STEP_SIZES)]
    return y_values, iterations


def test_time(algorithm_name: str, a: float, dim: int):
    stmt = f"{algorithm_name}(MathFunction(fun, {a}, {dim}), np.random.uniform(-100, 100, {dim}))"
    return timeit.timeit(stmt, globals=globals(), number=1)


def create_plot(path: str, data: Tuple[List[List[float]], List[int]], title: str, label_x: str = "", label_y: str = ""):
    plt.figure()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    for i, element in enumerate(data[1]):
        plt.plot(np.arange(element), np.array(data[0][i]), label=str(STEP_SIZES[i]))
    if (not len(data[1]) == 1):
        plt.legend(loc="upper right")
    plt.savefig(path)
    plt.clf()
