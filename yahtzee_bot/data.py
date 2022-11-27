import numpy as np


class CollectSampleExperiments():
    """
    Generates samples of parts played by the actions of the model

    Based on the rules on https://www.hasbro.com/common/instruct/yahtzee.pdf

    Parameters
    ----------
    n_games : int
        Number of games played in parallel
    n_dices : int
        Number of dices involved
    dice_max_value : int
        Maximum value of a dice
    n_boxes : int
        Number of boxes to fill with one for the yahtzee bonuses
    model_dice_1 : keras model
        Neural network that generates the best combination of dice to keep in the first round
    model_dice_2 : keras model
        Neural network that generates the best combination of dice to keep in the second round
    model_box : keras model
        Neural network that generates the best choice of checkbox

    Returns
    ----------
    history_model_dice_1 : List[List[(array, int, float)]]
        Sample states played by the model dice 1
    history_model_dice_2 : List[List[(array, int, float)]]
        Sample states played by the model dice 2
    history_model_box : List[List[((array, array), int, float)]]
        Sample states played by the model box
    """

    def __init__(self, n_games, n_dices, dice_max_value, n_boxes, model_dice_1, model_dice_2, model_box):
        self.n_games = n_games
        self.n_dices = n_dices
        self.dice_max_value = dice_max_value
        self.n_boxes = n_boxes
        self.model_dice_1 = model_dice_1
        self.model_dice_2 = model_dice_2
        self.model_box = model_box

    def initialize(self):
        self.value_box = np.zeros(
            (self.n_games, self.n_boxes), dtype=np.float32)
        self.is_box_checked = np.zeros(
            (self.n_games, self.n_boxes), dtype=np.int32)
        self.dice = np.zeros(
            (self.n_games, self.n_dices*self.dice_max_value), dtype=np.int32)
        self.history_model_dice_1 = [[] for _ in range(self.n_games)]
        self.history_model_dice_2 = [[] for _ in range(self.n_games)]
        self.history_model_box = [[] for _ in range(self.n_games)]

    def available_moves(self):
        """
        Generates an array of every possible move available, and an array determining if any box has been succefully reached


        - Determines which boxes can be checked

        - Determines which games have no real checkboxes and no Yahtzee

        - Determines which games have a Yahtzee or Joker bonus

        - Determines, for games where there is a Yahtzee bonus or a Joker, which boxes are checkable


        Returns
        ----------
        is_any_box_reached : array
            1 if a box has been reached, 0 else

        available_boxes : array
            Every box that the bot can check
        """
        self.available_boxes = np.zeros(
            (self.n_games, self.n_boxes), dtype=np.int32)

        self.n_identical_dices = np.zeros((self.n_games, 6), dtype=np.int32)
        for i in range(self.dice_max_value):
            self.n_identical_dices[:, i] = np.sum(
                self.dice[:, i+np.arange(0, self.n_dices)*self.dice_max_value], axis=1)

        # Determines which boxes can be checked

        # Upper Section

        # Aces
        self.available_boxes[:, 0] = (self.n_identical_dices[:, 0] > 0)

        # Twos
        self.available_boxes[:, 1] = (self.n_identical_dices[:, 1] > 0)

        # Threes
        self.available_boxes[:, 2] = (self.n_identical_dices[:, 2] > 0)

        # Fours
        self.available_boxes[:, 3] = (self.n_identical_dices[:, 3] > 0)

        # Fives
        self.available_boxes[:, 4] = (self.n_identical_dices[:, 4] > 0)

        # Sixes
        self.available_boxes[:, 5] = (self.n_identical_dices[:, 5] > 0)

        # Lower Section

        # 3 of a kind
        self.available_boxes[:, 6] = (self.n_identical_dices > 2).any(axis=1)

        # 4 of a kind
        self.available_boxes[:, 7] = (self.n_identical_dices > 3).any(axis=1)

        # Full House
        self.available_boxes[:, 8] = (self.n_identical_dices == 3).any(
            axis=1) * (self.n_identical_dices == 2).any(axis=1)

        # Small Straight
        self.available_boxes[:, 9] = np.clip((self.n_identical_dices[:, :3] > 0).all(axis=1) + (
            self.n_identical_dices[:, 1:4] > 0).all(axis=1) + (self.n_identical_dices[:, 2:] > 0).all(axis=1), 0, 1)

        # Large Straight
        self.available_boxes[:, 10] = (self.n_identical_dices[:, :3] > 0).all(
            axis=1) + (self.n_identical_dices[:, 2:] > 0).all(axis=1)

        # Chance
        self.available_boxes[:, 11] = 1

        # Yahtzee
        self.available_boxes[:, 12] = (self.n_identical_dices == 5).any(axis=1)
        # Determines which games have no real checkboxes and no Yahtzee

        self.is_any_box_reached = np.ones(self.n_games, dtype=np.int32)

        available_boxes_mask = self.available_boxes*(1-self.is_box_checked)
        no_available_box = (
            self.available_boxes[:, 12] == 0)*(available_boxes_mask == 0).all(axis=1)

        self.is_any_box_reached[no_available_box] = 0

        available_boxes_mask[no_available_box] = (
            1-self.is_box_checked)[no_available_box]

        # Determines which games have a Yahtzee or Joker bonus
        # In this case, the boxes [Full House, Small Straight, Large Straight] are added to available_boxes,
        # only if the upper section of the value of the dice is not empty

        self.is_yahtzee_bonus = np.zeros(self.n_games, dtype=np.int32)
        self.is_joker = np.zeros(self.n_games, dtype=np.int32)

        self.is_yahtzee_bonus[(self.available_boxes[:, 12]*self.is_box_checked[:, 12]
                               * (self.value_box[:, 12] > 0)).astype(bool)] = 1
        self.is_joker[(self.available_boxes[:, 12] *
                       self.is_box_checked[:, 12]).astype(bool)] = 1

        # Add self.is_yahtzee bonus to the reward function

        # Then, we modify the available_boxes array according to the rules
        # You need to score in the appropriate box in the Upper Section
        # If not possible, you need to score any available box in the Lower Section box
        # If not possible, you need to score any available box, but no point can be obtain

        is_first_joker = np.zeros(self.n_games, dtype=bool)
        is_second_joker = np.zeros(self.n_games, dtype=bool)
        is_third_joker = np.zeros(self.n_games, dtype=bool)

        is_first_joker[(self.is_box_checked[self.is_joker, np.argmax(
            self.dice[self.is_joker], axis=1)]).all()] = 1
        is_second_joker[(self.is_box_checked[self.is_joker -
                                             is_first_joker, 6:12]).all()] = 1
        is_third_joker = (self.is_joker - is_second_joker -
                          is_first_joker).astype(bool)

        # Determines, for games where there is a Yahtzee bonus or a Joker, which boxes are checkable

        self.is_any_box_reached[is_third_joker] = 1

        self.available_boxes[is_first_joker, np.argmax(
            self.dice[is_first_joker], axis=1)] = 1
        self.available_boxes[(1-self.is_box_checked)
                             [is_second_joker, 6:12]] = 1
        self.available_boxes[(1-self.is_box_checked)[is_third_joker]] = 1

    def determine_intermediate_reward(self):
        reward = np.zeros(self.n_games, dtype=np.float32)

        # Upper Section

        # Aces
        reward[self.decision == 0] = self.n_identical_dices[self.decision == 0, 0]

        # Twos
        reward[self.decision == 1] = self.n_identical_dices[self.decision == 1, 1]*2

        # Threes
        reward[self.decision == 2] = self.n_identical_dices[self.decision == 2, 2]*3

        # Fours
        reward[self.decision == 3] = self.n_identical_dices[self.decision == 3, 3]*4

        # Fives
        reward[self.decision == 4] = self.n_identical_dices[self.decision == 4, 4]*5

        # Sixes
        reward[self.decision == 5] = self.n_identical_dices[self.decision == 5, 5]*6

        # Lower Section

        # 3 of a kind
        reward[self.decision == 6] = np.sum(
            self.n_identical_dices[self.decision == 6]*np.arange(1, 7), axis=1)

        # 4 of a kind
        reward[self.decision == 7] = np.sum(
            self.n_identical_dices[self.decision == 7]*np.arange(1, 7), axis=1)

        # Full House
        reward[self.decision == 8] = 25

        # Small Straight
        reward[self.decision == 9] = 30

        # Large Straight
        reward[self.decision == 10] = 40

        # Chance
        reward[self.decision == 11] = np.sum(
            self.n_identical_dices[self.decision == 11]*np.arange(1, 7), axis=1)

        # Yahtzee
        reward[self.decision == 12] = 50

        # Yahtzee bonus
        reward[self.decision == 13] = 100

        self.value_box[np.arange(0, self.n_games), self.decision] += reward

        return reward

    def determine_final_reward(self):
        pass

    def generate_sample(self):
        pass
