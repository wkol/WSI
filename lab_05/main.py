from neural_network import NeuralNetwork
from math_function import MathFunction, neural_function
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
m = MathFunction(neural_function)
neurons = [20, 30, 35, 50, 80]
lrs = [0.01, 0.1, 0.001]
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_raw = np.linspace(-10, 10, 2000)
vfun = np.vectorize(m)
y_raw = vfun(x_raw)

x = np.linspace(-10, 10, 2000).reshape(2000, 1)
y = vfun(x).reshape(2000, 1)
print(x.min(), x.max(), y.min(), y.max())
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
     x, y, test_size=0.33, random_state=42)
rng = np.random.default_rng(1)
n = NeuralNetwork(rng, 50, 0.01, 0)
n.train(X_test, y_test, 50000)
a = n.predict(X_test)
a = y_scaler.inverse_transform(a)
x_temp = x_scaler.inverse_transform(X_test)
plt.plot(x_raw, y_raw, lw=0.5, label="actual")
plt.scatter(x_temp, a)
plt.savefig("ahsgdjasgd.png")
# print(a.max(), a.min())
# for lr in lrs:
#     plt.grid(True)
#     plt.plot(x_raw, y_raw, lw=0.5, label="actual")
#     for neuron in neurons:
#         rng = np.random.default_rng(1)
#         n = NeuralNetwork(rng, neuron, lr, 0)
#         n.train(X_train, y_train, 100000)
#         a = n.predict(X_test)
#         a = y_scaler.inverse_transform(a)
#         x_temp = x_scaler.inverse_transform(X_test)
#         plt.scatter(x_temp, a, label=f"n={neuron}, lr={lr}", s=0.5)
#     plt.legend(fontsize="xx-small")
#     plt.savefig(f"lr-{lr}.png")
#     plt.clf()
#     plt.cla()
# print(a)
