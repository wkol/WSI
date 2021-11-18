from utilities import test_time, test_step_size, create_plot
from function_optimization import FunctionOptimization
import os


DIMENSIONS = [10, 20]
A_PARAMETER = [1, 10, 100]


def main():
    i = 0
    for dim in DIMENSIONS:
        for a in A_PARAMETER:
            create_plot(os.getcwd() + f"/lab_01/chart/gradient/fig{i}.png",
                        test_step_size(FunctionOptimization.gradient_descent, a, dim),
                        "Wartości funkcji w poszczególnych iteracjach dla metody gradientu",
                        "Iteracja", "Wartość funkcji"
                        )
            create_plot(os.getcwd() + f"/lab_01/chart/newton_const/fig{i}.png",
                        test_step_size(FunctionOptimization.newton_constant_step, a, dim),
                        "Wartości funkcji w poszczególnych iteracjach dla metody Newtona",
                        "Iteracja", "Wartość funkcji"
                        )
            create_plot(os.getcwd() + f"/lab_01/chart/newton_backtracking/fig{i}.png",
                        test_step_size(FunctionOptimization.newton_backtracking_step, a, dim),
                        "Wartości funkcji w poszczególnych iteracjach dla metody Newtona(z nawrotami)",
                        "Iteracja", "Wartość funkcji"
                        )
            i += 1
    time_gradient = 0
    time_newton = 0
    time_backtracking = 0
    for dim in DIMENSIONS:
        for a in A_PARAMETER:
            time_gradient += test_time("FunctionOptimization.gradient_descent", a, dim)
            time_newton += test_time("FunctionOptimization.newton_constant_step", a, dim)
            time_backtracking += test_time("FunctionOptimization.newton_backtracking_step", a, dim)
    avg_gradient = time_gradient / (len(DIMENSIONS) * len(A_PARAMETER))
    avg_newton = time_newton / (len(DIMENSIONS) * len(A_PARAMETER))
    avg_backtracking = time_backtracking / (len(DIMENSIONS) * len(A_PARAMETER))
    print(f"avg time gradient: {avg_gradient}")
    print(f"avg time newton(without backtracking): {avg_newton}")
    print(f"avg time newton(with backtracking): {avg_backtracking}")


if __name__ == "__main__":
    main()
