import cvxopt
import numpy as np
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, kernel_type: str, **kwargs) -> None:
        self.kernel_type = kernel_type
        self.deg = kwargs.get("deg", 2)
        self.r = kwargs.get("r", 1)
        self.gamma = kwargs.get("gamma", 0.01)
        self.c = kwargs.get("c", 12.0)

    def kernel_function(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.kernel_type == 'linear':
            return np.dot(a, b.T)
        if self.kernel_type == "polynomial":
            return (self.gamma * np.dot(a, b.T) + self.r) ** self.deg
        if self.kernel_type == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(a - b) ** 2.0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.n = x.shape[0]

        # cvxopt takes problems in form of 1/2 * x.T * P * x + q.T * x
        # in our case x is alpha
        # Fill P matrix which equals to yj * yi * (xi · xj)
        p = np.empty((self.n, self.n))
        kernel_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                kernel_matrix[i, j] = self.kernel_function(x[i], x[j])
        p = matrix(np.outer(y, y) * kernel_matrix)
        # Fill q matrix full of -1
        q = matrix(-np.ones((self.n, 1)))

        # First constraint G * ai <= h in our case constraint is  C > ai > 0
        G = matrix(np.vstack((-np.eye(self.n), np.eye(self.n))))
        h = matrix(np.hstack((np.zeros(self.n), np.ones(self.n) * self.c)))

        # Second constraint A * ai = b corresponds in or case to
        # Sum of(ai * yi) = 0
        A = matrix(y.reshape((1, self.n)), tc="d")  # In that case A equals to array of yi
        b = matrix(0.0)  # b equals 0

        solved_alphas = solvers.qp(p, q, G, h, A, b)
        self.alphas = np.array(solved_alphas["x"]).reshape(self.n)  # Get solved value
        firstSV = np.where(self.alphas > 1e-4)  # get first supporting vector
        if firstSV[0].size == 0:
            firstSV = 0
        else:
            firstSV = firstSV[0][0]

        sum = np.sum([self.alphas[i] * self.y[i] * self.kernel_function(self.x[i], self.x[firstSV]) for i in range(self.n)])
        self.b = sum - self.y[firstSV]

    # Predict class based on sign of decision function
    def decision(self, u: np.ndarray) -> int:
        sum = np.sum([self.alphas[i] * self.y[i] * self.kernel_function(self.x[i], u) for i in range(self.n)])
        if sum - self.b >= 0:
            return 1
        return -1

    # Predict multiple points
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return np.array([self.decision(x) for x in x_test])
