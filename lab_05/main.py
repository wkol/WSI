from neural_network import NeuralNetwork
from math_function import MathFunction, neural_function
from functions import FunctionCollection
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SAMPLES = 16000


def main():
    m = MathFunction(neural_function)
    neurons = [20, 40, 60, 80, 100, 120]
    lrs = [0.5, 0.1, 0.01]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_raw = np.linspace(-40, 40, SAMPLES)
    y_raw = m(x_raw)
    train_x = x_scaler.fit_transform(x_raw.reshape(-1, 1))
    train_y = y_scaler.fit_transform(y_raw.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=0.33, random_state=42)
    for lr in lrs:
        plt.grid(True)
        plt.plot(x_raw, y_raw, lw=0.5, label="actual")
        for neuron in neurons:
            rng = np.random.default_rng(1)
            n = NeuralNetwork(rng, 2, neuron, FunctionCollection.sigmoid,
                              FunctionCollection.loss_MSE, lr)
            n.train(X_train.reshape(1, -1), y_train.reshape(1, -1), 100000, 200)
            a = n.predict(X_test.reshape(1, -1))
            a = y_scaler.inverse_transform(a.reshape(-1, 1))
            x_temp = x_scaler.inverse_transform(X_test)
            plt.scatter(x_temp.flatten(), a,
                        label=f"n={neuron}, lr={lr}", s=0.5)
        plt.legend(fontsize="xx-small")
        plt.savefig(f"lr-{lr}.png")
        plt.clf()
        plt.cla()


if __name__ == "__main__":
    main()
