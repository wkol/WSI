from typing import Callable, Tuple
import numpy as np
from math_function import MathFunction


class Layer:
    def __init__(self, rng: np.random.Generator, weight_shape: Tuple[int]) -> None:
        self.rng = rng
        self.initialize_weights(weight_shape)

    def feed_forward(self, input: np.ndarray,  activation_fun: Callable) -> Tuple[np.ndarray]:
        """
        Allows to feedforward through layer.
        Returns summed_input (w[i] * x[i]) and layer output
        """
        summed_layer = np.dot(self.weights, input) + self.bias
        return summed_layer, activation_fun(summed_layer)

    def update_weights(self, weight_gradient: np.ndarray, bias_gradient: np.ndarray, learning_rate: float) -> None:
        """
        Updates layer's weights and biases based on given gradients
        """
        self.weights = self.weights - weight_gradient * learning_rate
        self.bias = self.bias - bias_gradient * learning_rate

    def initialize_weights(self, weight_shape: Tuple[int]) -> None:
        """
        Random weight initialization
        """
        self.weights = self.rng.standard_normal(weight_shape)
        self.bias = self.rng.standard_normal((weight_shape[0], 1))

    def backpropagate(self, delta: np.ndarray, input: np.ndarray, activation_derivative: Callable) -> np.ndarray:
        """
        Allows to backpropagate through layer.
        """
        return np.dot(self.weights.T, delta) * activation_derivative(input)


class NeuralNetwork:
    def __init__(self, rng: np.random.Generator, layers: int,
                 hidden_neurons: int,
                 activation_function: MathFunction,
                 loss_function: MathFunction,
                 learning_rate: float = 0.1) -> None:
        self.rng = rng
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.initialize_layers(layers, hidden_neurons)

    def train(self, input: np.ndarray, actual: np.ndarray, epoch_num: int, batch_size: int) -> None:
        """
        Trains neural network with backpropagation and batch_size learning
        """
        n = input.shape[1]
        for _ in range(epoch_num):
            for batch_i in range(0, n, batch_size):
                input_batch = input[0][batch_i:batch_i +
                                       batch_size].reshape(1, -1)
                actual_batch = actual[0][batch_i:batch_i +
                                         batch_size].reshape(1, -1)
                inputs, outputs = self.feed_forward(input_batch)
                dw, db = self.backpropagate(inputs, outputs, actual_batch)
                for i, layer in enumerate(self.layers):
                    layer.update_weights(dw[i], db[i], self.learning_rate)

    def feed_forward(self, train_data: np.ndarray) -> Tuple[np.ndarray]:
        """
        Feeds forward through all layers then returns input and
        output for each layer
        """
        inputs = []
        outputs = [train_data]  # Ouput from input layer
        for layer in self.layers:
            inp, out = layer.feed_forward(
                outputs[-1], self.activation_function)
            inputs.append(inp)
            outputs.append(out)
        return inputs, outputs

    def backpropagate(self, inputs: np.ndarray, outputs: np.ndarray, actual: np.ndarray) -> Tuple[np.ndarray]:
        """
        Backpropagates through all layers then returns gradients for each layer's
        weights and biases
        """
        loss_derivative = self.loss_function.derivative(outputs[-1], actual)
        deltas = [loss_derivative *
                  self.activation_function.derivative(inputs[-1])]
        for i in reversed(range(len(self.layers) - 1)):
            delta = self.layers[i+1].backpropagate(
                deltas[-1], inputs[i], self.activation_function.derivative)
            # stores deltas in order to reuse them in the next layer
            deltas.append(delta)
        deltas = deltas[::-1]
        batch_size = actual.shape[1]
        weightes_gradients = []
        biases_gradients = []
        for i, delta in enumerate(deltas):
            weightes_gradients.append(
                np.dot(delta, outputs[i].T) / float(batch_size))  # Gradient of weights
            biases_gradients.append(np.dot(delta, np.ones(
                (batch_size, 1))) / float(batch_size))  # Gradient of biases
        return weightes_gradients, biases_gradients

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predicts value output for given input
        """
        return self.feed_forward(input)[1][-1]

    def initialize_layers(self, layers_num: int, neurons: int) -> None:
        """
        Initializes each of layers
        """
        layers = [1] + [neurons] * layers_num + [1]
        self.layers = [Layer(self.rng, (layers[i+1], layers[i]))
                       for i in range(len(layers) - 1)]
