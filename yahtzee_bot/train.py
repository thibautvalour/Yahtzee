import numpy as np
import tensorflow

from data import CollectSampleExperiments
from network import DiceReRoll, BoxChoice
from ppo_loss import PPO


class Unit():
    def __init__(self, hidden, hidden_value, hidden_policy, output_policy):
        self.hidden = hidden
        self.hidden_value = hidden_value
        self.hidden_policy = hidden_policy
        self.output_policy = output_policy


def train(n_ite, n_games, unit_dice_1, unit_dice_2, unit_box):
    model_dice_1 = DiceReRoll(unit_dice_1)
    model_dice_2 = DiceReRoll(unit_dice_2)
    model_box = BoxChoice(unit_box)
    collect_sample = CollectSampleExperiments(
        n_games, 5, 6, 13, 50., 200., model_dice_1, model_dice_2, model_box)

    for _ in range(n_ite):
        collect_sample.generate_sample()
        PPO(collect_sample)

    return collect_sample.model_dice_1, collect_sample.model_dice_2, collect_sample.model_box
