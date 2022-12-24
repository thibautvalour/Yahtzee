"""Yahtzee, the board game!
This python program will simulate a game of Yahtzee againt a bot
"""
import sys
import os

from yahtzee_game.hand import Hand
from yahtzee_game.scoreboard import ScoreBoard
from yahtzee_bot.data import CollectSampleExperiments


def Main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
YAHTZEE
Welcome to the game. To begin, simply follow the instructions on the screen.
To exit, press [Ctrl+C]""")

    # Begin by instantiating the hands and scoreboards
    hand = Hand(5, 6)
    scoreboard_player = ScoreBoard()

    bot = CollectSampleExperiments(1, None, None, None, 5, 6, 13, 50., 0.9)
    bot.initialize_inference(path="./yahtzee_bot/weights")

    # We play the 13 rounds to fill the grid
    for round in range(13):
        print("player 1 to play")
        hand.throw()
        hand.show_hand()
        hand.re_roll()
        scoreboard_player.select_scoring(hand)
        scoreboard_player.show_scoreboard_points()

        input("\nPress any key to continue")
        os.system('cls' if os.name == 'nt' else 'clear')

        print("player 2 (bot) to play")
        bot.show_dice(step=1)
        bot.displays_dice_rerolled()
        bot.show_dice(step=2)
        bot.displays_dice_rerolled()
        bot.show_dice(step=3)
        bot.show_scoreboard_points(round)

        input("\nPress any key to continue")
        os.system('cls' if os.name == 'nt' else 'clear')

    score_player_1 = sum(scoreboard_player.get_scoreboard_points().values())
    score_player_2 = int(bot.determine_final_reward()[0])
    if score_player_1 == score_player_2:
        print(f"It's a draw! You both have {score_player_1} points!")
    else:
        worst_score, best_score = sorted([score_player_1, score_player_2])
        winner = [score_player_1, score_player_2].index(best_score) + 1
        if winner == 1:
            print(f"\nCongratulations human player !")
            print(
                f"You finished the game with {best_score} points while bot has {worst_score} points\n")
        if winner == 2:
            print(f"\nCongratulations bot !")
            print(
                f"You finished the game with {best_score} points while the human has {worst_score} points\n")


if __name__ == '__main__':
    try:
        Main()
    except KeyboardInterrupt:
        print("\nExiting...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
