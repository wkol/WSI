from typing import Tuple
import numpy as np
import numdifftools as nd
from math_function import MathFunction, fun

class FunctionOptimization:

    @staticmethod
    def gradient_descent(fun: MathFunction, starting_point: np.ndarray, step_size: float = 0.001, precision: float = 1e-04) -> Tuple[np.ndarray, int]:
        gradient = nd.Gradient(fun)
        current_point = starting_point
        iters = 0
        while True:
            iters+=1
            gradient_at_point = gradient(current_point)
            next_point = current_point - step_size * gradient_at_point
            points_diff = np.abs(next_point - current_point)
            current_point = next_point
            if np.all(np.abs(gradient_at_point) <= precision) or np.all(points_diff <= precision):
                return current_point, iters

    @staticmethod            
    def newton_constant_step(fun: MathFunction, starting_point: np.ndarray, step_size: float = 0.001, precision: float = 1e-04) -> Tuple[np.ndarray, int]:
        current_point = starting_point
        hessian = nd.Hessian(fun)
        gradient = nd.Gradient(fun)
        iters = 0
        while True:
            iters+=1
            gradient_at_point = gradient(current_point)
            hessian_at_pointt_inv = np.linalg.inv(hessian(current_point)) 
            next_point = current_point - step_size * np.matmul(hessian_at_pointt_inv, gradient_at_point)
            points_diff = np.abs(next_point - current_point)
            current_point = next_point
            if np.all(np.abs(points_diff) <= precision):
                return current_point, iters

    @staticmethod
    def newton_backtracking(fun: MathFunction, starting_point: np.ndarray, precision: float = 1e-04) -> Tuple[np.ndarray, int]:
        current_point = starting_point
        hessian = nd.Hessian(fun)
        gradient = nd.Gradient(fun)
        iters = 0
        while True:
            iters+=1
            gradient_at_point = gradient(current_point)
            hessian_at_pointt_inv = np.linalg.inv(hessian(current_point)) 
            direction = np.matmul(hessian_at_pointt_inv, gradient_at_point)
            step_size = FunctionOptimization._backtracking_line_search(fun, current_point, gradient, direction)
            next_point = current_point - direction * step_size
            points_diff = np.abs(next_point - current_point)
            current_point = next_point
            if np.all(np.abs(points_diff) <= precision):
                return current_point, iters
    
    @staticmethod
    def _backtracking_line_search(fun: MathFunction, point: np.ndarray, gradient: nd.Gradient, direction: np.ndarray) -> float:
        t = 1
        alpha = 0.3
        beta = 0.6
        while fun(point - t * direction) > fun(point) + alpha *  t * np.dot(np.transpose(gradient(point)), -direction):
            t *= beta
        return t
