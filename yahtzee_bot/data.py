from copy import copy
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
    normalization_boxes_reward : float
        Normalization constant for the boxes that helps the neural network by having almost only values between 0 and 1
    normalization_final_reward : float
        Normalization constant for the final reward that helps the neural network by having almost only values between 0 and 1
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

    def __init__(self, n_games, n_dices, dice_max_value, n_boxes, normalization_boxes_reward, normalization_final_reward, model_dice_1, model_dice_2, model_box):
        self.n_games = n_games
        self.n_dices = n_dices
        self.dice_max_value = dice_max_value
        self.n_boxes = n_boxes
        self.normalization_boxes_reward = normalization_boxes_reward
        self.normalization_final_reward = normalization_final_reward
        self.model_dice_1 = model_dice_1
        self.model_dice_2 = model_dice_2
        self.model_box = model_box

    def initialize(self):
        self.value_box = np.zeros(
            (self.n_games, self.n_boxes+1), dtype=np.float32)
        # +1 because of the Yahtzee bonus
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
        self.available_boxes[:, 10] = (self.n_identical_dices[:, :4] > 0).all(
            axis=1) + (self.n_identical_dices[:, 1:] > 0).all(axis=1)

        # Chance
        self.available_boxes[:, 11] = 1

        # Yahtzee
        self.available_boxes[:, 12] = (self.n_identical_dices == 5).any(axis=1)
        # Determines which games have no real checkboxes and no Yahtzee

        self.is_any_box_reached = np.ones(self.n_games, dtype=np.int32)

        available_boxes_mask = self.available_boxes*(1-self.is_box_checked)
        no_available_box = (
            self.available_boxes[:, 12] == 0)*(available_boxes_mask == 0).all(axis=1)

        self.available_boxes[no_available_box] = 1 - \
            self.is_box_checked[no_available_box]

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

        no_joker_and_available_actions = np.clip(
            (1-no_available_box-self.is_joker), 0, 1).astype(bool)

        self.available_boxes[no_joker_and_available_actions] = available_boxes_mask[no_joker_and_available_actions]

        # Add self.is_yahtzee bonus to the reward function

        # Then, we modify the available_boxes array according to the rules
        # You need to score in the appropriate box in the Upper Section
        # If not possible, you need to score any available box in the Lower Section box
        # If not possible, you need to score any available box, but no point can be obtain

        is_first_joker = np.zeros(self.n_games, dtype=bool)
        is_second_joker = np.zeros(self.n_games, dtype=bool)
        is_third_joker = np.zeros(self.n_games, dtype=bool)

        is_first_joker[self.is_joker.astype(bool)] = (1-self.is_box_checked[self.is_joker.astype(
            bool), np.argmax(self.dice[self.is_joker.astype(bool)], axis=1)]).astype(bool)

        is_second_joker[(self.is_joker - is_first_joker).astype(bool)] = (
            self.is_box_checked[(self.is_joker - is_first_joker).astype(bool), 6:12] == 0).any(axis=1)

        is_third_joker = (self.is_joker - is_second_joker -
                          is_first_joker).astype(bool)

        # Determines, for games where there is a Yahtzee bonus or a Joker, which boxes are checkable

        self.is_any_box_reached[is_third_joker] = 0

        self.available_boxes[is_first_joker] = 0
        self.available_boxes[is_first_joker, np.argmax(
            self.dice[is_first_joker.astype(bool)], axis=1)] = 1

        self.available_boxes[is_second_joker] = 0
        self.available_boxes[is_second_joker, 6:12] = (
            1-self.is_box_checked)[is_second_joker, 6:12]

        self.available_boxes[is_third_joker] = 0
        self.available_boxes[is_third_joker] = (
            1-self.is_box_checked)[is_third_joker].astype(bool)

    def determine_intermediate_reward(self):

        # Upper Section

        # Aces
        self.value_box[self.decision == 0, 0] = self.n_identical_dices[self.decision ==
                                                                       0, 0] * self.is_any_box_reached[self.decision == 0]

        # Twos
        self.value_box[self.decision == 1, 1] = self.n_identical_dices[self.decision ==
                                                                       1, 1] * self.is_any_box_reached[self.decision == 1]*2

        # Threes
        self.value_box[self.decision == 2, 2] = self.n_identical_dices[self.decision ==
                                                                       2, 2] * self.is_any_box_reached[self.decision == 2]*3

        # Fours
        self.value_box[self.decision == 3, 3] = self.n_identical_dices[self.decision ==
                                                                       3, 3] * self.is_any_box_reached[self.decision == 3]*4

        # Fives
        self.value_box[self.decision == 4, 4] = self.n_identical_dices[self.decision ==
                                                                       4, 4] * self.is_any_box_reached[self.decision == 4]*5

        # Sixes
        self.value_box[self.decision == 5, 5] = self.n_identical_dices[self.decision ==
                                                                       5, 5] * self.is_any_box_reached[self.decision == 5]*6

        # Lower Section

        # 3 of a kind
        self.value_box[self.decision == 6, 6] = np.sum(
            self.n_identical_dices[self.decision == 6]*np.arange(1, 7), axis=1) * self.is_any_box_reached[self.decision == 6]

        # 4 of a kind
        self.value_box[self.decision == 7, 7] = np.sum(
            self.n_identical_dices[self.decision == 7]*np.arange(1, 7), axis=1) * self.is_any_box_reached[self.decision == 7]

        # Full House
        self.value_box[self.decision == 8, 8] = 25 * \
            self.is_any_box_reached[self.decision == 8]

        # Small Straight
        self.value_box[self.decision == 9, 9] = 30 * \
            self.is_any_box_reached[self.decision == 9]

        # Large Straight
        self.value_box[self.decision == 10, 10] = 40 * \
            self.is_any_box_reached[self.decision == 10]

        # Chance
        self.value_box[self.decision == 11, 11] = np.sum(
            self.n_identical_dices[self.decision == 11]*np.arange(1, 7), axis=1) * self.is_any_box_reached[self.decision == 11]

        # Yahtzee
        self.value_box[self.decision == 12, 12] = 50 * \
            self.is_any_box_reached[self.decision == 12]

        # Yahtzee bonus
        self.value_box[self.is_yahtzee_bonus == 1, 13] += 100

        for i in range(self.n_games):
            self.is_box_checked[i, self.decision[i]] = 1

    def determine_final_reward(self):
        score_upper_section = np.sum(self.value_box[:, :6], axis=1)
        bonus = (score_upper_section >= 63)*35
        total_score_upper_section = score_upper_section + bonus
        total_score_lower_section = np.sum(self.value_box[:, 6:], axis=1)
        grand_total = total_score_upper_section + total_score_lower_section
        return grand_total

    def update_dice(self, dice):
        self.dice = np.zeros(
            (self.n_games, self.n_dices*self.dice_max_value), dtype=np.int32)
        for i in range(self.n_games):
            for j in range(self.n_dices):
                self.dice[i, dice[i, j]+6*j] = 1

    def average_reward_statistic(self):
        reward = self.normalization_final_reward*np.array([self.history_model_box[i][0][2] for i in range(self.n_games)], dtype=np.float32)
        mean = np.mean(reward)
        ecart = 1.96 * np.std(reward)/np.sqrt(self.n_games)
        return np.around([mean-ecart,mean, mean+ecart]).astype(np.int32)

    def generate_sample(self):
        self.initialize()

        for _ in range(self.n_boxes):
            roll_dice = np.random.randint(
                0, self.dice_max_value, (self.n_games, self.n_dices))
            self.update_dice(roll_dice)

            # First dice roll
            states = np.concatenate(
                [self.is_box_checked, self.value_box/self.normalization_boxes_reward, self.dice], axis=1)
            outputs = self.model_dice_1(states)[0].numpy()
            outputs_reweighted = outputs/outputs.sum(axis=1, keepdims=True)
            retained_dice = np.zeros(
                (self.n_games, self.n_dices), dtype=np.uint8)
            for i in range(self.n_games):
                dice_combinaison = np.random.choice(
                    np.arange(2**self.n_dices), p=outputs_reweighted[i])

                self.history_model_dice_1[i] += copy(
                    [[states[i], dice_combinaison, 0]])

                retained_dice[i] = np.unpackbits(np.uint8(dice_combinaison))[
                    8-self.n_dices:]
            new_dices = np.random.randint(
                0, self.dice_max_value, (self.n_games, self.n_dices))
            updated_dices = roll_dice*retained_dice+new_dices*(1-retained_dice)
            self.update_dice(updated_dices)

            # Second dice roll

            states = np.concatenate(
                [self.is_box_checked, self.value_box/self.normalization_boxes_reward, self.dice], axis=1)
            outputs = self.model_dice_2(states)[0].numpy()
            outputs_reweighted = outputs/outputs.sum(axis=1, keepdims=True)
            retained_dice = np.zeros(
                (self.n_games, self.n_dices), dtype=np.uint8)
            for i in range(self.n_games):
                dice_combinaison = np.random.choice(
                    np.arange(2**self.n_dices), p=outputs_reweighted[i])

                self.history_model_dice_2[i] += copy(
                    [[states[i], dice_combinaison, 0]])

                retained_dice[i] = np.unpackbits(np.uint8(dice_combinaison))[
                    8-self.n_dices:]
            new_dices = np.random.randint(
                0, self.dice_max_value, (self.n_games, self.n_dices))
            updated_dices = roll_dice*retained_dice+new_dices*(1-retained_dice)
            self.update_dice(updated_dices)

            # Box to check

            self.available_moves()
            states = [np.concatenate([self.is_box_checked, self.value_box/self.normalization_boxes_reward,
                                     self.dice, self.available_boxes], axis=1), self.available_boxes.astype(np.float32)]

            outputs = self.model_box(states)[0].numpy()
            outputs_reweighted = outputs/outputs.sum(axis=1, keepdims=True)
            self.decision = np.zeros(self.n_games, dtype=np.int32)
            for i in range(self.n_games):
                self.decision[i] = np.random.choice(
                    np.arange(self.n_boxes), p=outputs_reweighted[i])

                self.history_model_box[i] += copy(
                    [[(states[0][i], states[1][i]), self.decision[i], 0]])

            self.determine_intermediate_reward()

        grand_total = self.determine_final_reward()

        for i in range(self.n_games):
            for j in range(self.n_boxes):
                reward = grand_total[i]/self.normalization_final_reward
                self.history_model_dice_1[i][j][2] = copy(reward)
                self.history_model_dice_2[i][j][2] = copy(reward)
                self.history_model_box[i][j][2] = copy(reward)
