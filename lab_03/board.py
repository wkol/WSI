

from typing import List, Optional
from move import Move
from constants import Cell, cells_sign
import numpy as np


class Board:
    def __init__(self, x_size, y_size) -> None:
        self._x_size = x_size
        self._y_size = y_size
        assert x_size == y_size
        self.current_board = np.full((y_size, x_size), Cell.EMPTY_CELL)
        self.heuristic_board = self.generate_heurestic_board(x_size, y_size)


    @staticmethod
    def generate_heurestic_board(x_size, y_size) -> List[List[int]]:
        heuristic_board = np.full((y_size, x_size), 2.0)
        heuristic_board += np.identity(x_size)
        heuristic_board += np.rot90(np.identity(x_size))
        return heuristic_board

    def __str__(self) -> str:
        board_str = ""
        for y in range(self._y_size):
            if y != 0:
                board_str += "-" * (self._x_size + 2) + "\n"
            for x in range(self._x_size):
                board_str += cells_sign[self.current_board[y][x]]
                if x != self._x_size - 1:
                    board_str += "|"
            board_str += "\n"
        return board_str


    def make_move(self, move: Move, cell: Cell) -> None:
        self.current_board[move.y_cord][move.x_cord] = cell

    def get_legal_moves(self) -> List[Move]:
        legal_moves = []
        for i in range(self._y_size):
            for j in range(self._x_size):
                if (self.current_board[i][j] == Cell.EMPTY_CELL):
                    legal_moves.append(Move(j, i))
        return legal_moves
    
    def is_legal_move(self, x, y) -> bool:
        if x not in range(0, self._x_size) or y not in range(0, self._y_size):
            return False
        if not self.current_board[y][x] == Cell.EMPTY_CELL:
            return False
        return True

    def wins(self) -> Optional[Cell]:
        for row in self.current_board:
            if np.all(row[0] == row) and row[0] != Cell.EMPTY_CELL:
                return row[0]
        for row in np.rot90(self.current_board):
            if np.all(row[0] == row) and row[0] != Cell.EMPTY_CELL:
                return row[0]
        if self._x_size == self._y_size:
            if np.all(self.current_board.diagonal() == self.current_board[0][0]) and self.current_board[0][0] != Cell.EMPTY_CELL:
                return self.current_board[0][0]
            if np.all(np.fliplr(self.current_board).diagonal() == self.current_board[0][self._x_size-1]) and self.current_board[0][self._x_size-1] != Cell.EMPTY_CELL:
                return self.current_board[0][self._x_size-1]
        return None

    def is_empty(self):
        return np.all(self.current_board == Cell.EMPTY_CELL)

    def end(self) -> bool:
        if self.wins() is not None:
            return True
        if self.get_legal_moves() == []:
            return True
        return False

