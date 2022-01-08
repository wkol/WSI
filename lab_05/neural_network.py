from typing import Callable, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.numeric import indices
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def activation_function( x: np.ndarray) -> np.ndarray:
    return 1 / (1.0 + np.exp(-x))

def activation_derivative(x: np.ndarray):
    return activation_function(x) * (1 - activation_function(x))

class Layer:
    def __init__(self, rng: np.random.Generator, weight_shape: Tuple[int]) -> None:
        self.rng = rng
        self.output = None
        self.activation_input = None
        self.initialize_weights(weight_shape)

    def feedforward(self, input: np.ndarray,  activation_fun: Callable):
        self.activation_input = np.dot(self.weights, input) + self.bias
        self.output = activation_fun(self.activation_input)
        return self.activation_input, self.output

    def update_weights(self, weight_gradient: np.ndarray, bias_gradient: np.ndarray, learning_rate: float):
        self.weights = self.weights - weight_gradient * learning_rate
        self.bias = self.bias - bias_gradient * learning_rate

    def initialize_weights(self, weight_shape):
        self.weights = self.rng.standard_normal(weight_shape)
        self.bias = self.rng.standard_normal((weight_shape[0], 1))

    def backpropagate(self, deltas: np.ndarray, input: np.ndarray):
        sp = activation_derivative(input)
        delta = np.dot(self.weights.T, deltas[-1])
        delta = delta * sp
        return delta

class NeuralNetwork:
    def __init__(self, rng: np.random.Generator, layers: List[int],
                 learning_rate: float = 0.1, epochs: int = 10000) -> None:
        self.rng = rng
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.initialize_layers(layers)

    def train(self, input: np.ndarray, actual: np.ndarray, epoch_num: int, batches: int):
        n = input.shape[1]
        for _ in range(epoch_num):
            for batch_i in range(0, n, batches):
                input_batch = input[0][batch_i:batch_i+batches].reshape(1, -1)
                actual_batch = actual[0][batch_i:batch_i+batches].reshape(1, -1)
                inputs, outputs = self.feed_forward(input_batch)
                dw, db = self.backpropagate(inputs, outputs, actual_batch)
                [layer.update_weights(dw[i], db[i], self.learning_rate) for i, layer in enumerate(self.layers)]

    def feed_forward(self, train_data: np.ndarray):
        inputs = []
        outputs = [train_data]
        for layer in self.layers:
            inp, out = layer.feedforward(outputs[-1], activation_function)
            inputs.append(inp)
            outputs.append(out)
        return inputs, outputs

    def backpropagate(self,inputs, outputs, actual: np.ndarray): # Backpropagation for all hideen layers
        loss_derivative = self.loss_derivative(outputs[-1], actual)
        deltas = [loss_derivative * activation_derivative(inputs[-1])]
        for i in reversed(range(len(self.layers) - 1)):
            dl = self.layers[i+1].backpropagate(deltas, inputs[i])
            deltas.append(dl)
        deltas = deltas[::-1]
        batch_size = actual.shape[1]
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
        dw = [d.dot(outputs[i].T)/float(batch_size) for i,d in enumerate(deltas)]
        return dw, db


    def predict(self, input: np.ndarray):
        return self.feed_forward(input)[1][-1]

    def loss_function(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.square(actual-predicted) / 2.0

    def loss_derivative(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return predicted - actual

    def initialize_layers(self, layers: List[int]) -> None:
        self.layers = [Layer(self.rng, (layers[i+1], layers[i])) for i in range(len(layers) - 1)]
