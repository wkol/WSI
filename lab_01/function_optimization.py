from typing import List, Tuple
import numpy as np
import numdifftools as nd
from math_function import MathFunction

MAX_ITERS = 1000


class FunctionOptimization:
    @staticmethod
    def gradient_descent(fun: MathFunction, starting_point: np.ndarray, step_size: float = 0.001, values: List[float] = None, precision: float = 1e-03) -> Tuple[np.ndarray, int]:
        if values is None:
            values = []
        current_point = starting_point
        gradient = nd.Gradient(fun)
        iters = 0
        while iters < MAX_ITERS:
            iters += 1
            values.append(fun(current_point))
            gradient_at_point = gradient(current_point)
            next_point = current_point - step_size * gradient_at_point
            points_diff = np.abs(next_point - current_point)
            current_point = next_point
            if np.all(np.abs(points_diff) <= precision):
                return current_point, iters
        return None, iters

    @staticmethod
    def newton_constant_step(fun: MathFunction, starting_point: np.ndarray, step_size: float = 0.001, values: List[float] = None, precision: float = 1e-03) -> Tuple[np.ndarray, int]:
        if values is None:
            values = []
        current_point = starting_point
        hessian = nd.Hessian(fun)
        gradient = nd.Gradient(fun)
        iters = 0
        while iters < MAX_ITERS:
            iters += 1
            values.append(fun(current_point))
            gradient_at_point = gradient(current_point)
            hessian_at_pointt_inv = np.linalg.inv(hessian(current_point))
            next_point = current_point - step_size * np.matmul(hessian_at_pointt_inv, gradient_at_point)
            points_diff = np.abs(next_point - current_point)
            current_point = next_point
            if np.all(np.abs(points_diff) <= precision):
                return current_point, iters
        return None, iters

    @staticmethod
    def newton_backtracking_step(fun: MathFunction, starting_point: np.ndarray, values: List[float] = None, precision: float = 1e-03) -> Tuple[np.ndarray, int]:
        if values is None:
            values = []
        current_point = starting_point
        hessian = nd.Hessian(fun)
        gradient = nd.Gradient(fun)
        iters = 0
        step_size = 1
        while iters < MAX_ITERS:
            iters += 1
            values.append(fun(current_point))
            gradient_at_point = gradient(current_point)
            hessian_at_pointt_inv = np.linalg.inv(hessian(current_point)) 
            direction = np.matmul(hessian_at_pointt_inv, gradient_at_point)
            step_size = FunctionOptimization._backtracking_line_search(fun, current_point, step_size, gradient, direction)
            next_point = current_point - direction * step_size
            points_diff = np.abs(next_point - current_point)
            current_point = next_point
            if np.all(np.abs(points_diff) <= precision):
                return current_point, iters
        return None, iters

    @staticmethod
    def _backtracking_line_search(fun: MathFunction, point: np.ndarray, step: float, gradient: nd.Gradient, direction: np.ndarray) -> float:
        alpha = 0.3
        beta = 0.6
        while fun(point - step * direction) > fun(point) + alpha * step * np.dot(np.transpose(gradient(point)), -direction):
            step *= beta
        return step
