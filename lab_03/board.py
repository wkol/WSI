

from typing import List, Optional
from move import Move
from constants import Cell, cells_sign
import numpy as np


class Board:
    def __init__(self, size) -> None:
        self.size = size
        self.current_board = np.full((size, size), Cell.EMPTY_CELL)
        self.heuristic_board = self.generate_heurestic_board(size)

    @staticmethod
    def generate_heurestic_board(size) -> List[List[int]]:
        """
        Static method which generates heuristic board.
        Each cell corresponds to the amount of the winning combination
        which includes that cell
        """
        heuristic_board = np.full((size, size), 2.0)
        heuristic_board += np.identity(size)
        heuristic_board += np.rot90(np.identity(size))
        return heuristic_board

    def __str__(self) -> str:
        board_str = ""
        for y in range(self.size):
            if y != 0:
                board_str += "-" * (self.size * 2 - 1) + "\n"
            for x in range(self.size):
                board_str += cells_sign[self.current_board[y][x]]
                if x != self.size - 1:
                    board_str += "|"
            board_str += "\n"
        return board_str

    def make_move(self, move: Move, cell: Cell) -> None:
        self.current_board[move.y_cord][move.x_cord] = cell

    def get_legal_moves(self) -> List[Move]:
        """
        Method returning list of all legal moves
        """
        legal_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.current_board[i][j] == Cell.EMPTY_CELL):
                    legal_moves.append(Move(j, i))
        return legal_moves

    def is_legal_move(self, x: int, y: int) -> bool:
        """
        Method checking if the move at given coordinates is legal
        """
        if x not in range(0, self.size) or y not in range(0, self.size):
            return False
        if not self.current_board[y][x] == Cell.EMPTY_CELL:
            return False
        return True

    def wins(self) -> Optional[Cell]:
        """
        Method checking if game is won by any of the players.
        Return None in case when game is neither end nor won 
        """
        for row in self.current_board:
            if np.all(row[0] == row) and row[0] != Cell.EMPTY_CELL:
                return row[0]
        for row in np.rot90(self.current_board):
            if np.all(row[0] == row) and row[0] != Cell.EMPTY_CELL:
                return row[0]
        if self.size == self.size:
            if np.all(self.current_board.diagonal() == self.current_board[0][0]) and self.current_board[0][0] != Cell.EMPTY_CELL:
                return self.current_board[0][0]
            if np.all(np.fliplr(self.current_board).diagonal() == self.current_board[0][self.size-1]) and self.current_board[0][self.size-1] != Cell.EMPTY_CELL:
                return self.current_board[0][self.size-1]
        return None

    def is_empty(self) -> bool:
        return np.all(self.current_board == Cell.EMPTY_CELL)

    def end(self) -> bool:
        """
        Method which returns True or False wheather game has
        ended (one of the player won or draw) or not
        """
        if self.wins() is not None:
            return True
        if self.get_legal_moves() == []:
            return True
        return False
