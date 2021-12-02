from constants import Cell
from abc import ABC, abstractclassmethod
from board import Board
from math import inf
import random
from move import Move
from constants import cells_sign

class Player(ABC):
    def __init__(self, cell_type: Cell) -> None:
        assert Cell(cell_type) in [Cell.O_CELL, Cell.X_CELL]
        self.cell_type = Cell(cell_type)
         
    @abstractclassmethod
    def make_move(self, board: Board) -> Move:
        pass


class ComputerPlayer(Player):
    def __init__(self, cell_type: Cell, depth: int = 4) -> None:
        super().__init__(cell_type)
        self.max_depth = depth

    def make_move(self, board: Board) -> Move:
        if board.is_empty():
            return Move(random.randint(0, board._x_size - 1),
                        random.randint(0, board._y_size - 1))
        return self.minimax(board, self.cell_type, 0)

    def minimax(self, board: Board, cell_type: Cell, depth: int) -> Move:
        score = -inf if cell_type == self.cell_type else inf  # Select init score based on the player's sign  
        best_move = Move(-1, -1, score)
        if depth == self.max_depth or board.end():
            if self.cell_type  == cell_type:
                return Move(-1, -1, self.evaluate(board, cell_type) - depth)
            return Move(-1, -1, self.evaluate(board, cell_type) + depth)
        for legal_move in board.get_legal_moves():
            board.current_board[legal_move.y_cord][legal_move.x_cord] = cell_type
            move = self.minimax(board, Cell(-cell_type.value), depth + 1)
            board.current_board[legal_move.y_cord][legal_move.x_cord] = Cell.EMPTY_CELL # Undo move to avoid deepCopying of the Board object
            move.y_cord, move.x_cord = legal_move.y_cord, legal_move.x_cord
            if cell_type == self.cell_type:
                best_move = max(best_move, move, key=lambda move: move.score)  # Computer player - maximize score
            else:
                best_move = min(best_move, move, key=lambda move: move.score)  # Other player - minimazie score
        return best_move

    def evaluate(self, board: Board, cell_type: Cell) -> int:
        if board.wins() == self.cell_type:
            return 10000
        if board.wins() == Cell(-self.cell_type.value):
            return -10000
        else:
            return self.evalute_heura(board, cell_type)

    def evalute_heura(self, board: Board, cell_type: Cell) -> int:
        result = 0
        for i in range(len(board.current_board)):
            for j in range(len(board.current_board[i])):
                if self.cell_type == cell_type:
                    if cell_type == board.current_board[i][j]:
                        result += board.heuristic_board[i][j]
                    elif board.current_board[i][j] != Cell.EMPTY_CELL:
                        result -= board.heuristic_board[i][j]
                else:
                    if cell_type == board.current_board[i][j]:
                        result -= board.heuristic_board[i][j]
                    elif board.current_board[i][j] != Cell.EMPTY_CELL:
                        result += board.heuristic_board[i][j]
        return result


class HumanPlayer(Player):
    def __init__(self, cell_type: int) -> None:
        super().__init__(cell_type)

    def make_move(self, board: Board) -> Move:
        y = input(f"Select a row to put {cells_sign[self.cell_type]} [1,2..{board._y_size}]: ")
        x = input(f"Select a collumn to put {cells_sign[self.cell_type]} [1,2..{board._x_size}]: ")
        while not x.isdigit() or not y.isdigit() or not board.is_legal_move(int(x) - 1, int(y) - 1):
            print("Invalid input. Select again")
            y = input(f"Select a row [1,2..{board._y_size}]: ")
            x = input(f"Select a collumn [1,2..{board._x_size}]: ")
        return Move(int(x) - 1, int(y) - 1)
