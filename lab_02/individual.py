from typing import Generator
from math_function import MathFunction
import numpy as np

class Individual:
    def __init__(self, params: np.ndarray, sigma: float) -> None:
        self.params = params
        self.fitness = None
        self.sigma = sigma

    def calculate_fitness(self, fun: MathFunction) -> None:
        self.fitness = fun(self.params)
    
    def mutate(self, rng: np.random.Generator) -> None:
        self.params =  self.params + rng.multivariate_normal(mean=np.full(len(self.params), 0), cov=np.identity(10)) * self.sigma