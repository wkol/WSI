from neural_network import NeuralNetwork
from math_function import MathFunction, neural_function
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(-1,1))
m = MathFunction(lambda x: x*x)
x = np.linspace(-5, 5, 400).reshape(400, 1)
vfun = np.vectorize(m)
y = vfun(x).reshape(400, 1)
print(x.min(), x.max(), y.min(), y.max())
x = scaler.fit_transform(x)
y = vfun(x).reshape(400, 1)

rng = np.random.default_rng()
n = NeuralNetwork(rng, 16, 0.01, 0)
n.train(x[100:], y[100:], 100000)
a = n.predict(x)
print(a)
plt.plot(x, y, label="actual")
plt.plot(x, a, label="predicted")
plt.savefig("0.1 10000 8.png")
# print(a)
