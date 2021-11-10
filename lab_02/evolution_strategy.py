from individual import Individual
import numpy as np
from math_function import MathFunction, fun_evolution, fun
from typing import Dict, List, Any, Callable
from population import Population

class EvolutionStrategy:
    
    @staticmethod
    def mi_lambda_es(starting_point: np.ndarray, rng: np.random.Generator, mi_param: int, lambda_param: int, fun: Callable, start_sigma: float, tau: float, max_iters: int, adapt: bool = True) -> Population:
        population = Population(Individual(starting_point, start_sigma), mi_param, lambda_param, rng, fun)
        iters = 0
        while iters < max_iters:
            population.population = population.recombine()
            if(adapt):
                population.update_self_adapt_sigma(tau)
            else:
                population.update_sigma()
            population.mutation()
            population.calculate_fitness()
            population.population = population.sort_population()
            population.population = population.succesion()
            population.update_centroid()
            iters += 1
        return population.centroid.params

