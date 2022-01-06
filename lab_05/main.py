from neural_network import NeuralNetwork
from math_function import MathFunction, neural_function
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
m = MathFunction(neural_function)
x = np.linspace(-1, 1, 400).reshape(400, 1)
vfun = np.vectorize(m)
y = vfun(x).reshape(400, 1)
print(x.min(), x.max(), y.min(), y.max())
x = x_scaler.fit_transform(x)
y_scaled = vfun(x).reshape(400, 1)
y = y_scaler.fit_transform(y)
rng = np.random.default_rng()
n = NeuralNetwork(rng, 16, 0.02, 0)
n.train(x[100:], y[100:], 10000)
a = n.predict(x)
print(a.max(), a.min())
x = x_scaler.inverse_transform(x)
a = y_scaler.inverse_transform(a)
y = y_scaler.inverse_transform(y)
plt.plot(x, y, label="actual")
plt.plot(x, a, label=f"predicted n = {8}")
    
print(a)
plt.legend()
plt.savefig("0.1 10000 stag1.png")
# print(a)
