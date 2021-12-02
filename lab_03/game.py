from board import Board
from player import Player
from typing import List
from constants import cells_sign, clear_screen
import random
import time

class Game:
    def __init__(self, players: List[Player], size: int, debug: bool) -> None:
        self.players = players
        self.dim = size
        self.board = Board(size)
        self._debug = debug

    def clean_board(self) -> None:
        """
        Method performing cleaning of the board
        """
        self.board = Board(self.dim)

    def swap_player(self, current_player: Player) -> Player:
        """
        Method which returns oppent of the current player
        """
        return self.players[0] if self.players[1].cell_type.value == current_player.cell_type.value else self.players[1]

    def play(self) -> None:
        """
        Main game method which plays a game
        """
        clear_screen()
        current_player = random.choice(self.players)
        print(f"{cells_sign[current_player.cell_type]} goes first")
        time.sleep(1)
        while not self.board.end():
            print(self.board)
            print(f"Player with {cells_sign[current_player.cell_type]} sign turn:")
            self.board.make_move(current_player.make_move(self.board), current_player.cell_type)
            current_player = self.swap_player(current_player)
            if not self._debug:
                clear_screen()
        print(self.board)
        if self.board.wins() is None:
            print("Game ended as a draw")
        else:
            print(f"Game won by {cells_sign[self.board.wins()]} player")