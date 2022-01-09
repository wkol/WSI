import numpy as np
import math_function


def activation_sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Activation sigmoid function
    """
    return 1 / (1.0 + np.exp(-x))


def activation_derivative(x: np.ndarray):
    """
    Derivative of sigmoid activation function
    """
    return activation_sigmoid_function(x) * (1 - activation_sigmoid_function(x))


def loss_function(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Loss function - mean squared error
    """
    return np.square(actual-predicted) / 2.0


def loss_derivative(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Derivative of loss mean squared error function
    """
    return predicted - actual


class FunctionCollection:
    sigmoid = math_function.MathFunction(activation_sigmoid_function,
                                         activation_derivative)
    loss_MSE = math_function.MathFunction(loss_function, loss_derivative)
