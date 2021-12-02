from board import Board
from player import Player
from typing import List
from constants import cells_sign, clear_screen
import random
import time

class Game:
    def __init__(self, players: List[Player], x_dim: int, y_dim: int) -> None:
        self.players = players
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.board = Board(x_dim, y_dim)

    def clean_board(self):
        self.board = Board(self.x_dim, self.y_dim)

    def swap_player(self, current_player: Player):
        return self.players[0] if self.players[1].cell_type.value == current_player.cell_type.value else self.players[1]

    def play(self):
        current_player = self.players[1]# random.choice(self.players)
        print(self.board)
        print(f"{cells_sign[current_player.cell_type]} goes first")
        while not self.board.end():
            # time.sleep(1)
            # clear_screen()
            print(f"Player with {cells_sign[current_player.cell_type]} sign turn:")
            self.board.make_move(current_player.make_move(self.board), current_player.cell_type)
            print(self.board)
            current_player = self.swap_player(current_player)
        if self.board.wins() is None:
            print("Game ended as a draw")
        else:
            raise RuntimeError
            print(f"Game won by {cells_sign[self.board.wins()]} player")