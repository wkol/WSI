from typing import Callable, Tuple, List
from matplotlib import pyplot as plt
import numpy as np
from evolution_strategy import EvolutionStrategy
from math_function import MathFunction, fun_evolution
import timeit

seed = 0

def seed_generator(update: bool) -> np.random.Generator:
    global seed
    if update:
        seed += 1
    return np.random.default_rng(seed)


def create_data(mi_lambda: Tuple[int, int], sigmas: List[float], strategy: EvolutionStrategy.mi_lambda_es, fun: MathFunction) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = seed_generator(False)
    starting_point = rng.uniform(-100, 100, 10)
    tau = 10 ** (-1/2)
    y_vals_lmr = []
    y_vals_sa = []
    for i,sigma in enumerate(sigmas):
        y_vals_lmr.append(np.mean([strategy(starting_point, fun, seed_generator(True), mi_lambda[0], mi_lambda[1], sigma, tau=tau, adapt=False)[1] for _ in range(5)], axis=0))
        y_vals_sa.append(np.mean([strategy(starting_point, fun, seed_generator(True), mi_lambda[0], mi_lambda[1], sigma, tau=tau, adapt=True)[1] for _ in range(5)], axis=0))
    return y_vals_lmr, y_vals_sa


def test_time(algorithm: EvolutionStrategy.mi_lambda_es) -> Tuple[float, float]:
    rng = seed_generator(True)
    starting_point = rng.uniform(-100, 100, 10)
    tau = 10 ** (-1/2)
    mi_lambda = (4, 20)
    sigma = 0.01
    fun = MathFunction(fun_evolution, 10)
    time_lmr = []
    time_sa =[]
    for _ in range(5):
        start_lmr = timeit.default_timer()
        algorithm(starting_point, fun, rng, mi_lambda[0], mi_lambda[1], sigma, tau, 2000, False)
        end_lmr = timeit.default_timer()
        time_lmr.append(end_lmr - start_lmr)
        start_sa = timeit.default_timer()
        algorithm(starting_point, fun, rng, mi_lambda[0], mi_lambda[1], sigma, tau, 2000, True)
        end_sa = timeit.default_timer()
        time_sa.append(end_sa - start_sa)
    return sum(time_lmr) / 5, sum(time_sa) / 5


def create_plot(path: str, data: Tuple[List[np.ndarray], List[np.ndarray]], title: str, label_x: str = "", label_y: str = "", labels: List[str] = None) -> None:
    if labels is None:
        labels = ["0.01", "0.1", "1"]
    fig, (ax1, ax2) = plt.subplots(2)
    # plot lmr method
    fig.set_size_inches(18.5, 10.5)
    ax1.set_title("LMR method")
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    ax1.grid(True)
    for i, element in enumerate(data[0]):
        ax1.plot(np.arange(element.size), element, label=labels[i])
    
    # plot sa method
    ax2.set_title("Self-adaptation method")
    ax2.set_xlabel(label_x)
    ax2.set_ylabel(label_y)
    ax2.grid(True)
    for i, element in enumerate(data[1]):
        ax2.plot(np.arange(element.size), element, label=labels[i])
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.legend(loc="upper right", title="Starting sigma value")
    ax2.legend(loc="upper right", title="Starting sigma value")
    fig.subplots_adjust(hspace=.5)
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    fig.clf()
