from individual import Individual
import numpy as np
from typing import Dict, List, Any, Callable, Tuple
from population import Population

class EvolutionStrategy:
    
    @staticmethod
    def mi_lambda_es(starting_point: np.ndarray, fun: Callable, rng: np.random.Generator, mi_param: int, lambda_param: int, start_sigma: float, tau: float, max_iters: int = 2000, adapt: bool = True) -> Tuple[np.ndarray, List[float]]:
        population = Population(Individual(starting_point, start_sigma), mi_param, lambda_param, rng, fun)
        population.centroid.calculate_fitness(fun)
        y_vals = []
        iters = 0
        while iters < max_iters:
            y_vals.append(population.centroid.fitness)
            population.population = population.recombine()
            if(adapt):
                population.update_self_adapt_sigma(tau)
            else:
                population.update_sigma(tau)
            population.mutation()
            population.calculate_fitness()
            population.population = population.sort_population()
            population.population = population.succesion()
            population.update_centroid()
            iters += 1
        return population.centroid.params, np.array(y_vals)

