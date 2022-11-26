import numpy as np


class CollectSampleExperiments():
    """
    Generates samples of parts played by the actions of the model

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
        # Based on the rules on https://www.hasbro.com/common/instruct/yahtzee.pdf
        # without the wadzee bonus
        available_boxes = np.zeros(
            (self.n_games, self.n_boxes), dtype=np.int32)

        n_identical_dices = np.zeros((self.n_games, 6), dtype=np.int32)
        for i in range(self.dice_max_value):
            n_identical_dices[:, i] = np.sum(
                self.dice[:, i+np.arange(0, self.n_dices)*self.dice_max_value], axis=1)

        # Upper Section

        # Aces
        available_boxes[:, 0] = (n_identical_dices[:, 0] > 0)

        # Twos
        available_boxes[:, 1] = (n_identical_dices[:, 1] > 0)

        # Threes
        available_boxes[:, 2] = (n_identical_dices[:, 2] > 0)

        # Fours
        available_boxes[:, 3] = (n_identical_dices[:, 3] > 0)

        # Fives
        available_boxes[:, 4] = (n_identical_dices[:, 4] > 0)

        # Sixes
        available_boxes[:, 5] = (n_identical_dices[:, 5] > 0)

        # Upper Section

        # 3 of a kind
        available_boxes[:, 6] = (n_identical_dices > 2).any(axis=1)

        # 4 of a kind
        available_boxes[:, 7] = (n_identical_dices > 3).any(axis=1)

        # Full House
        available_boxes[:, 8] = (n_identical_dices == 3).any(
            axis=1) * (n_identical_dices == 2).any(axis=1)

        # Small Straight
        available_boxes[:, 9] = np.clip((n_identical_dices[:, :3] > 0).all(axis=1) + (
            n_identical_dices[:, 1:4] > 0).all(axis=1) + (n_identical_dices[:, 2:] > 0).all(axis=1), 0, 1)

        # Large Straight
        available_boxes[:, 10] = (n_identical_dices[:, :3] > 0).all(
            axis=1) + (n_identical_dices[:, 2:] > 0).all(axis=1)

        # Chance
        available_boxes[:, 11] = 1

        # Yahtzee
        available_boxes[:, 12] = (n_identical_dices == 5).any(axis=1)

        available_boxes_mask = available_boxes*(1-self.is_box_checked)

        # If the player has no available box and no Yahtzee

        no_available_box = (available_boxes[:, 12]==0)*(available_boxes_mask == 0).all(axis=1)
        available_boxes_mask[no_available_box] = (1-self.is_box_checked)[no_available_box]

        # First case : Joker
        # In this case, the boxes [Full House, Small Straight, Large Straight] are added to available_boxes,
        # only if the upper section of the value of the dice is not empty
        yahtzee_indexes_joker = available_boxes[((available_boxes_mask == 0).all(axis=1)*available_boxes[:, 12]*self.is_box_checked[:, 12]*(self.value_box[:, 12] == 0)).astype(bool)]
        available_boxes[yahtzee_indexes_joker, 8:10] = 1

        # Second case : Yahtzee Bonus
        yahtzee_indexes_bonus = available_boxes[(available_boxes[:, 12]*self.is_box_checked[:, 12]*(self.value_box[:, 12] > 0)).astype(bool)]
        available_boxes[yahtzee_indexes_bonus, 13] += 1
        available_boxes[yahtzee_indexes_bonus, :13] = 1

        # We reapply mask
        available_boxes_mask = available_boxes*(1-self.is_box_checked)
        no_available_box = (available_boxes[:, 12]==0)*(available_boxes_mask == 0).all(axis=1)
        available_boxes_mask[no_available_box] = (1-self.is_box_checked)[no_available_box]

        return available_boxes_mask

    def determine_intermediate_reward(self):
        pass

    def determine_final_reward(self):
        pass

    def generate_sample(self):
        pass
