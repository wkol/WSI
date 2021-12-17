from numpy import vectorize
from numpy.core.function_base import linspace
from utils import get_data_from_file, grade_quality
from svm import SVM
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    gamma_params = [0.001, 0.01, 0.1]
    x_dataset, y_dataset = get_data_from_file("winequality-red.csv")
    vfunc = np.vectorize(grade_quality)
    c_params = np.linspace(0.001, 100)
    y_dataset = vfunc(y_dataset)
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0) 
    accuracy = [[], [], []]
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Impact of c parameter on accuracy")
    plt.grid(True)
    for i, g in enumerate(gamma_params):
        for c in c_params:
            svm = SVM("rbf", gamma=g, c=c)
            svm.fit(x_train, y_train)
            accuracy[i].append(accuracy_score(y_test, svm.predict(x_test)))
        plt.plot(c_params, accuracy[i], label=f"gamma={g}")
        plt.legend(loc="lower right")
    plt.savefig("plot_rbf.png")
    plt.clf()
    accuracy = [[], [], []]
    gamma_params = [0.1, 10, 200]
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Impact of c parameter on accuracy")
    plt.grid(True)
    c = linspace(0.0001, 1, 50)
    for i, g in enumerate(gamma_params):
        for c in c_params:
            svm = SVM("polynomial", gamma=g, deg=2, r=3, c=c)
            svm.fit(x_train, y_train)
            accuracy[i].append(accuracy_score(y_test, svm.predict(x_test)))
        plt.plot(c_params, accuracy[i], label=f"gamma={g}")
        plt.legend(loc="lower right")
    plt.savefig("plot_polynomial.png")


if __name__ == "__main__":
    main()
