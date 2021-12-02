from enum import Enum
import platform
import os

def clear_screen() -> None:
    if platform.system() == "Linux":
        os.system("clear")
    elif platform.systemq() == "Windows":
        os.system("cls")

class Cell(Enum):
    X_CELL = -1
    O_CELL = 1
    EMPTY_CELL = 2

cells_sign = {Cell.EMPTY_CELL: " ", Cell.O_CELL: "O", Cell.X_CELL: "X"}