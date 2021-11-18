import numpy as np
from typing import List

from numpy.core.numeric import identity
from individual import Individual
from math_function import MathFunction

class Population:
    
    def __init__(self, centroid: Individual, mi_param: int, lambda_param: int, rng: np.random.Generator, fun: MathFunction) -> None:
        self.population = None
        self.mi_param = mi_param
        self.lambda_param = lambda_param
        self.fun = fun
        self.rng = rng
        self.centroid = centroid
        self.dim = centroid.params.size


    def __repr__(self) -> str:
        return f"Centroid: {self.centroid.params}, Fitness: {self.fun(self.centroid.params)}, "


    def sort_population(self) -> List[Individual]:
        return sorted(self.population, key= lambda ind: ind.fitness)


    def update_centroid(self) -> None:
        self.centroid.params = np.mean([ind.params for ind in self.population], 0)
        self.centroid.sigma = sum([ind.sigma for ind in self.population]) / self.mi_param
        self.centroid.calculate_fitness(self.fun)


    def calculate_fitness(self) -> None:
        for element in self.population:
            element.calculate_fitness(self.fun)


    def succesion(self) -> List[Individual]:
        return self.population[:self.mi_param]


    def recombine(self) -> List[Individual]:
        return [Individual(np.copy(self.centroid.params), self.centroid.sigma) for _ in range(self.lambda_param)]


    def update_sigma(self, tau) -> None:
        sigma = self.centroid.sigma * np.exp(tau * self.rng.normal(0.0, 1.0))
        for element in self.population:
            element.sigma = sigma


    def update_self_adapt_sigma(self, tau) -> None:
        for element in self.population:
            sigma = self.centroid.sigma * np.exp(tau * self.rng.normal(loc=0, scale=1.0))
            element.sigma = sigma


    def mutation(self) -> None:
        for element in self.population:
            element.mutate(self.rng)