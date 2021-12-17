import numpy as np
from typing import List, Tuple


def get_data_from_file(file_name: str) -> Tuple[np.ndarray]:
    data = np.genfromtxt(file_name, delimiter=";", skip_header=1)
    return data[:, :-1], data[:, -1]


def grade_quality(quality: int) -> int:
    return 1 if quality > 5 else -1
