from utilities import create_plot, create_data, test_time
from evolution_strategy import EvolutionStrategy
from math_function import fun_sphere, fun_evolution, MathFunction
import os
import numpy as np

MI_LAMBDA_PARAMS = [(4, 20), (10, 20), (20, 100), (50, 200)]
SIGMA_PARAMS = [0.01, 0.1, 1]
FUNS = [MathFunction(fun_sphere, 1, 10), MathFunction(fun_evolution, 10)]


def main():
    i = 0
    for func in FUNS:
        for mi_lambda in MI_LAMBDA_PARAMS:
            create_plot(os.getcwd() + f"/chart/fig{i}.png",
            create_data(mi_lambda, SIGMA_PARAMS, EvolutionStrategy.mi_lambda_es, func),
            "Fitness function in each iteration of the evolution strategy method")
            i+=1
    print(test_time(EvolutionStrategy.mi_lambda_es))

if __name__ == "__main__":
    main()