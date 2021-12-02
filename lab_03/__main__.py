from typing import List, Tuple
from constants import Cell, clear_screen
from game import Game
from player import ComputerPlayer, HumanPlayer, Player


def get_players() -> List[Player]:
    player_1 = input("Select type of the first player (X): (Human, Computer) ")
    while player_1 not in ["Human", "Computer"]:
        player_1 = input("Invalid input Select again: (Human, Computer) ")
    player_2 = input("Select type of the second player (O): (Human, Computer) ")
    while player_2 not in ["Human", "Computer"]:
        player_2 = input("Invalid input Select again: (Human, Computer) ")
    if player_1 == "Computer":
        player_1 = ComputerPlayer(Cell.X_CELL, 8)
    else:
        player_1 = HumanPlayer(Cell.X_CELL)
    if player_2 == "Computer":
        player_2 = ComputerPlayer(Cell.O_CELL, 8)
    else:
        player_2 = HumanPlayer(Cell.O_CELL)
    return [player_1, player_2]


def get_board_sizes() -> Tuple[int, int]:
    x_size = input("Select number of collumns on the board (>2): ")
    while not x_size.isdigit() and int(x_size) < 3:
        x_size = input("Invalid input. Select a positive, greater than 2 number: ")
    y_size = input("Select number of rows on the board (>2): ")
    while not x_size.isdigit() and int(x_size) < 3:
        y_size = input("Invalid input. Select a positive, greater than 2 number: ")
    return int(x_size), int(y_size)


def main():
    print("Tic tac toe game")
    x_size, y_size = get_board_sizes()
    players = get_players()
    clear_screen()
    game = Game(players, x_size, y_size)
    game.play()


if __name__ == "__main__":
    for _ in range(20000):
        x_size, y_size = get_board_sizes()
        players = get_players()
        game = Game(players, x_size, y_size)
        game.play()
