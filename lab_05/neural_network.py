import numpy as np


class NeuralNetwork:
    def __init__(self, rng: np.random.Generator, neurons: int,
                 learning_rate: float = 5, bias: float = 0) -> None:
        self.rng = rng
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.bias = bias

    def train(self, input: np.ndarray, actual: np.ndarray, epoch_num: int):
        self.initialize_weights(input, actual)
        errors = []
        for _ in range(epoch_num):
            # hidden_layer_sum = np.dot(input, self.inner_weights) + self.w0
            # hidden_output = self.activation_function(hidden_layer_sum)

            # output_layer_sum = np.dot(hidden_output,self.outer_weights) + self.v0
            # predicted_y = self.activation_function(output_layer_sum)
            
            # loss = self.loss_function(predicted_y, actual)
            # errors.append(loss)

            # delta2 = (predicted_y - actual) * predicted_y * (1 - predicted_y)
            # outer_weights_gradient = np.dot(hidden_output.T, delta2)
            # self.outer_weights = self.outer_weights - outer_weights_gradient * self.learning_rate
            # self.v0 = self.v0 - np.sum(delta2, axis=0, keepdims=True) * self.learning_rate

            # delta1 = np.dot(delta2, self.outer_weights.T) * hidden_output * (1 - hidden_output)
            # inner_weights_gradient = np.dot(input.T, delta1)
            # self.inner_weights = self.inner_weights - inner_weights_gradient * self.learning_rate
            # self.w0 = self.w0 - np.sum(delta1, axis=0, keepdims=True) * self.learning_rate
        # return errors    
            # # Forward pass
            # Hidden layer
            hidden_sum = np.dot(input, self.inner_weights) + self.w0
            hidden_activation = self.activation_function(hidden_sum)

            # Output layer
            output_layer_input = hidden_activation.dot(self.outer_weights) + self.v0
            y_pred = self.activation_function(output_layer_input)
            # print(f"Loss: {loss}")
            # Backpropagation
            # Output layer
            gradient_output_layer = self.loss_derivative(y_pred, actual) * self.activation_derivative(output_layer_input)
            gradient_output_weights = hidden_activation.T.dot(gradient_output_layer)
            gradient_output_bias = np.sum(gradient_output_layer, axis=0, keepdims=True)

            # Hidden layer
            gradient_hidden_layer = gradient_output_layer.dot(self.outer_weights.T) * self.activation_derivative(hidden_sum)
            gradient_hidden_weights = input.T.dot(gradient_hidden_layer)
            gradient_hidden_bias = np.sum(gradient_hidden_layer, axis=0, keepdims=True)

            # Update weights
            self.outer_weights -= self.learning_rate * gradient_output_weights
            self.v0 -= self.learning_rate * gradient_output_bias
            self.inner_weights -= self.learning_rate * gradient_hidden_weights
            self.w0 -= self.learning_rate * gradient_hidden_bias
    def predict(self, input: np.ndarray):
        hidden_input = input.dot(self.inner_weights) + self.w0
        hidden_output = self.activation_function(hidden_input)
        output_layer_input = np.dot(hidden_output, self.outer_weights) + self.v0
        y_pred = self.activation_function(output_layer_input)
        return y_pred

    def loss_function(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.square(actual-predicted) / 2.0

    def loss_derivative(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return predicted - actual

    def activation_function(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, 0)

    def activation_derivative(self, x):
        return np.where(x >= 0, 1, 0)

    def initialize_weights(self, x: np.ndarray, y: np.ndarray):
        # limit = 1
        # self.inner_weights = np.random.uniform(-limit, limit, (1, self.neurons))
        # self.w0 = np.zeros((1, self.neurons))
        # # Output layer
        # limit = self.neurons ** -0.5
        # self.outer_weights = np.random.uniform(-limit, limit, (self.neurons, 1))
        # self.v0 = np.zeros((1, 1))
        #Weights input layer - hidden layer
        self.inner_weights = self.rng.standard_normal(size=(1, self.neurons))
        self.w0 = np.zeros(shape=(1, self.neurons))
        # Weights hidden layer - output layer
        self.outer_weights = self.rng.standard_normal(size=(self.neurons, 1))
        self.v0 = np.zeros(shape=(1, 1))
    
"""
Pick a random weights.
Optimize

"""

