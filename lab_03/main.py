from typing import List
from constants import Cell, clear_screen
from game import Game
from player import ComputerPlayer, HumanPlayer, Player
import argparse
import sys


def get_players(depth: int) -> List[Player]:
    player_1 = input("Select type of the first player (X): (Human, Computer) ")
    while player_1 not in ["Human", "Computer"]:
        player_1 = input("Invalid input Select again: (Human, Computer) ")
    player_2 = input("Select type of the second player (O): (Human, Computer) ")
    while player_2 not in ["Human", "Computer"]:
        player_2 = input("Invalid input Select again: (Human, Computer) ")
    if player_1 == "Computer":
        player_1 = ComputerPlayer(Cell.X_CELL, depth)
    else:
        player_1 = HumanPlayer(Cell.X_CELL)
    if player_2 == "Computer":
        player_2 = ComputerPlayer(Cell.O_CELL, depth)
    else:
        player_2 = HumanPlayer(Cell.O_CELL)
    return [player_1, player_2]


def get_board_sizes() -> int:
    size = input("Select size of the board (>2): ")
    while not size.isdigit() and int(size) < 3:
        size = input("Invalid input. Select a positive, greater than 2 number: ")
    return int(size)


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser("Tic tac toe game with minimax algorithm")
    parser.add_argument('--depth', default=8, type=int)
    depth = parser.parse_args().depth
    print("Tic tac toe game")
    size = get_board_sizes()
    players = get_players(depth)
    clear_screen()
    game = Game(players, size)
    end = False
    while not end:
        game.play()
        decision = input("Do you want to play another game (Y/N)")
        if decision == "N":
            end = True
        else:
            game.clean_board()


if __name__ == "__main__":
    main(sys.argv)
