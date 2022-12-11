"""Yahtzee, the board game!
This python program will simulate a game of Yahtzee
"""
import sys
import os

from hand import Hand
from scoreboard import ScoreBoard


def Main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
YAHTZEE
Welcome to the game. To begin, simply follow the instructions on the screen.
To exit, press [Ctrl+C]""")

    # Begin by instantiating the hands and scoreboards
    hand_1, hand_2 = Hand(5, 6), Hand(5, 6)
    scoreboard_player_1, scoreboard_player_2 = ScoreBoard(), ScoreBoard()

    # We play the 13 rounds to fill the grid
    for round in range(13):
        print("player 1 to play")
        hand_1.throw()
        hand_1.show_hand()
        hand_1.re_roll()
        scoreboard_player_1.select_scoring(hand_1)
        scoreboard_player_1.show_scoreboard_points()
        
        print("player 2 to play")
        hand_2.throw()
        hand_2.show_hand()
        hand_2.re_roll()
        scoreboard_player_2.select_scoring(hand_2)
        scoreboard_player_2.show_scoreboard_points()

        input("\nPress any key to continue")
        os.system('cls' if os.name == 'nt' else 'clear')


    score_player_1 = sum(scoreboard_player_1.get_scoreboard_points().values())
    score_player_2 = sum(scoreboard_player_2.get_scoreboard_points().values())
    if score_player_1 == score_player_2:
        print(f"It's a draw! You both have {score_player_1} points!")
    else:
        worst_score, best_score = sorted([score_player_1, score_player_2])
        winner = [score_player_1, score_player_2].index(best_score) + 1
        print(f"\nCongratulations to player {winner}!")
        print(f"You finished the game with {best_score} points while player 2 has {worst_score} points\n")

if __name__ == '__main__':
    try:
        Main()
    except KeyboardInterrupt:
        print("\nExiting...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)