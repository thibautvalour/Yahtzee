from copy import copy
import numpy as np
import tensorflow

from yahtzee_bot.hash import Hash
from yahtzee_bot.network import RLModel


class CollectSampleExperiments():
    """
    Generates samples of parts played by the actions of the model

    Based on the rules on https://www.hasbro.com/common/instruct/yahtzee.pdf

    Parameters
    ----------
    n_games : int
        Number of games played in parallel
    model_dice_1 : keras model
        Neural network that generates the best combination of dice to keep in the first round
    model_dice_2 : keras model
        Neural network that generates the best combination of dice to keep in the second round
    model_box : keras model
        Neural network that generates the best choice of checkbox
    n_dice : int
        Number of dice involved
    dice_max_value : int
        Maximum value of a dice
    n_boxes : int
        Number of boxes to fill with one for the yahtzee bonuses
    normalization_boxes_reward : float
        Normalization constant for the boxes that helps the neural network by having almost only values between 0 and 1
    gamma : float
        Constant that determines the reward discount

    Returns
    ----------
    history_model_dice_1 : List[List[(array, int, float)]]
        Sample states played by the model dice 1
    history_model_dice_2 : List[List[(array, int, float)]]
        Sample states played by the model dice 2
    history_model_box : List[List[((array, array), int, float)]]
        Sample states played by the model box
    """

    def __init__(self, n_games: int, model_dice_1, model_dice_2, model_box, n_dice: int = 5, dice_max_value: int = 6, n_boxes: int = 13, normalization_boxes_reward: float = 50., gamma: float = 0.9) -> None:
        self.n_games = n_games
        self.n_dice = n_dice
        self.dice_max_value = dice_max_value
        self.n_boxes = n_boxes
        self.normalization_boxes_reward = normalization_boxes_reward
        self.gamma = gamma
        self.model_dice_1 = model_dice_1
        self.model_dice_2 = model_dice_2
        self.model_box = model_box
        self.hash = Hash(self.dice_max_value)
        self.hash.initialize()

    def initialize(self):
        self.value_box = np.zeros(
            (self.n_games, self.n_boxes+1), dtype=np.float32)
        # +1 because of the Yahtzee bonus
        self.is_box_checked = np.zeros(
            (self.n_games, self.n_boxes), dtype=np.int32)
        self.dice = np.zeros(
            (self.n_games, self.n_dice*self.dice_max_value), dtype=np.int32)
        self.history_model_dice_1 = [[] for _ in range(self.n_games)]
        self.history_model_dice_2 = [[] for _ in range(self.n_games)]
        self.history_model_box = [[] for _ in range(self.n_games)]

    def update_dice(self, dice):
        """
        Update the value of the dice
        """
        self.dice = np.zeros(
            (self.n_games, self.n_dice*self.dice_max_value), dtype=np.int32)
        for i in range(self.n_games):
            for j in range(self.n_dice):
                self.dice[i, dice[i, j]+6*j] = 1

    def update_one_hot_dice(self):
        """
        Update the n_identical_dice and n_identical_dice_one_hot array when the dice array is changed
        """
        self.n_identical_dice = np.zeros(
            (self.n_games, self.dice_max_value), dtype=np.int32)
        self.n_identical_dice_one_hot = np.zeros(
            (self.n_games, self.dice_max_value**2), dtype=np.int32)
        for i in range(self.dice_max_value):
            self.n_identical_dice[:, i] = np.sum(
                self.dice[:, i+np.arange(0, self.n_dice)*self.dice_max_value], axis=1)
            self.n_identical_dice_one_hot[np.arange(
                self.n_games), 6*i+self.n_identical_dice[:, i]] = 1

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

        self.update_one_hot_dice()

        # Determines which boxes can be checked

        # Upper Section

        # Aces
        self.available_boxes[:, 0] = (self.n_identical_dice[:, 0] > 0)

        # Twos
        self.available_boxes[:, 1] = (self.n_identical_dice[:, 1] > 0)

        # Threes
        self.available_boxes[:, 2] = (self.n_identical_dice[:, 2] > 0)

        # Fours
        self.available_boxes[:, 3] = (self.n_identical_dice[:, 3] > 0)

        # Fives
        self.available_boxes[:, 4] = (self.n_identical_dice[:, 4] > 0)

        # Sixes
        self.available_boxes[:, 5] = (self.n_identical_dice[:, 5] > 0)

        # Lower Section

        # 3 of a kind
        self.available_boxes[:, 6] = (self.n_identical_dice > 2).any(axis=1)

        # 4 of a kind
        self.available_boxes[:, 7] = (self.n_identical_dice > 3).any(axis=1)

        # Full House
        self.available_boxes[:, 8] = (self.n_identical_dice == 3).any(
            axis=1) * (self.n_identical_dice == 2).any(axis=1)

        # Small Straight
        self.available_boxes[:, 9] = np.clip((self.n_identical_dice[:, 0:4] > 0).all(axis=1) + (
            self.n_identical_dice[:, 1:5] > 0).all(axis=1) + (self.n_identical_dice[:, 2:6] > 0).all(axis=1), 0, 1)

        # Large Straight
        self.available_boxes[:, 10] = (self.n_identical_dice[:, 0:5] > 0).all(
            axis=1) + (self.n_identical_dice[:, 1:6] > 0).all(axis=1)

        # Chance
        self.available_boxes[:, 11] = 1

        # Yahtzee
        self.available_boxes[:, 12] = (self.n_identical_dice == 5).any(axis=1)
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

    def determine_intermediate_reward(self, turn):
        """
        Increment the value_box function, and assign a reward
        """

        # Upper Section

        # Aces
        self.value_box[self.decision == 0, 0] = self.n_identical_dice[self.decision ==
                                                                      0, 0] * self.is_any_box_reached[self.decision == 0]
        self.reward[self.decision ==
                    0] += self.value_box[self.decision == 0, 0]

        # Twos
        self.value_box[self.decision == 1, 1] = self.n_identical_dice[self.decision ==
                                                                      1, 1] * self.is_any_box_reached[self.decision == 1]*2
        self.reward[self.decision ==
                    1] += self.value_box[self.decision == 1, 1]

        # Threes
        self.value_box[self.decision == 2, 2] = self.n_identical_dice[self.decision ==
                                                                      2, 2] * self.is_any_box_reached[self.decision == 2]*3
        self.reward[self.decision ==
                    2] += self.value_box[self.decision == 2, 2]

        # Fours
        self.value_box[self.decision == 3, 3] = self.n_identical_dice[self.decision ==
                                                                      3, 3] * self.is_any_box_reached[self.decision == 3]*4
        self.reward[self.decision ==
                    3] += self.value_box[self.decision == 3, 3]

        # Fives
        self.value_box[self.decision == 4, 4] = self.n_identical_dice[self.decision ==
                                                                      4, 4] * self.is_any_box_reached[self.decision == 4]*5
        self.reward[self.decision ==
                    4] += self.value_box[self.decision == 4, 4]

        # Sixes
        self.value_box[self.decision == 5, 5] = self.n_identical_dice[self.decision ==
                                                                      5, 5] * self.is_any_box_reached[self.decision == 5]*6
        self.reward[self.decision ==
                    5] += self.value_box[self.decision == 5, 5]

        # Lower Section

        # 3 of a kind
        self.value_box[self.decision == 6, 6] = np.sum(
            self.n_identical_dice[self.decision == 6]*np.arange(1, 7), axis=1) * self.is_any_box_reached[self.decision == 6]
        self.reward[self.decision ==
                    6] += self.value_box[self.decision == 6, 6]

        # 4 of a kind
        self.value_box[self.decision == 7, 7] = np.sum(
            self.n_identical_dice[self.decision == 7]*np.arange(1, 7), axis=1) * self.is_any_box_reached[self.decision == 7]
        self.reward[self.decision ==
                    7] += self.value_box[self.decision == 7, 7]

        # Full House
        self.value_box[self.decision == 8, 8] = 25 * \
            self.is_any_box_reached[self.decision == 8]
        self.reward[self.decision ==
                    8] += self.value_box[self.decision == 8, 8]

        # Small Straight
        self.value_box[self.decision == 9, 9] = 30 * \
            self.is_any_box_reached[self.decision == 9]
        self.reward[self.decision ==
                    9] += self.value_box[self.decision == 9, 9]

        # Large Straight
        self.value_box[self.decision == 10, 10] = 40 * \
            self.is_any_box_reached[self.decision == 10]
        self.reward[self.decision ==
                    10] += self.value_box[self.decision == 10, 10]

        # Chance
        self.value_box[self.decision == 11, 11] = np.sum(
            self.n_identical_dice[self.decision == 11]*np.arange(1, 7), axis=1) * self.is_any_box_reached[self.decision == 11]
        self.reward[self.decision ==
                    11] += self.value_box[self.decision == 11, 11]

        # Yahtzee
        self.value_box[self.decision == 12, 12] = 50 * \
            self.is_any_box_reached[self.decision == 12]
        self.reward[self.decision ==
                    12] += self.value_box[self.decision == 12, 12]

        # Yahtzee bonus
        self.value_box[self.is_yahtzee_bonus == 1, 13] += 100
        self.reward[self.is_yahtzee_bonus == 1] += 100

        for i in range(self.n_games):
            self.is_box_checked[i, self.decision[i]] = 1

        if turn == self.n_boxes-1:
            self.reward[self.value_box[:, :6].sum(axis=1) >= 63] += 35

    def determine_final_reward(self):
        """
        Determines the final reward of the game by summing the value_box array
        """
        score_upper_section = np.sum(self.value_box[:, :6], axis=1)
        bonus = (score_upper_section >= 63)*35
        total_score_upper_section = score_upper_section + bonus
        total_score_lower_section = np.sum(self.value_box[:, 6:], axis=1)
        grand_total = total_score_upper_section + total_score_lower_section
        return grand_total

    def average_reward_statistic(self):
        """
        Gives statistics of the sample of games played: confidence interval, average, extreme values
        """
        grand_total = self.determine_final_reward()
        mean = np.mean(grand_total)
        ecart = 1.96 * np.std(grand_total)/np.sqrt(self.n_games)
        return np.around([mean-ecart, mean, mean+ecart, grand_total.min(), grand_total.max()]).astype(np.int32)

    def generate_sample(self, mode="test"):
        """
        This function generates a sample game of Yahtzee.
        It initializes the game by setting the number of games, dice, and boxes, and normalizing the box rewards.
        It then generates a random roll of the dice and updates the state of the game based on this roll.
        This is done twice to simulate two rolls of the dice.
        After each roll, the model makes a decision on which dice combinations to keep, and which to reroll.
        Finally, the function makes a decision on which box to check based on the updated state of the game.
        """
        self.initialize()
        for turn in range(self.n_boxes):
            roll_dice = np.random.randint(
                0, self.dice_max_value, (self.n_games, self.n_dice))
            self.update_dice(roll_dice)

            # First dice roll

            self.update_one_hot_dice()

            states = np.concatenate([self.is_box_checked, self.value_box /
                                    self.normalization_boxes_reward, self.n_identical_dice_one_hot], axis=1)

            available_combinaisons = self.hash.hash_function(
                self.n_identical_dice)

            outputs = self.model_dice_1(
                [states, available_combinaisons])[0].numpy()
            outputs_reweighted = outputs/outputs.sum(axis=1, keepdims=True)

            for i in range(self.n_games):
                if mode == "test":
                    dice_combinaison = np.random.choice(
                        np.arange(462), p=outputs_reweighted[i])
                else:
                    dice_combinaison = np.argmax(outputs_reweighted[i])
                self.history_model_dice_1[i] += copy(
                    [[(states[i], available_combinaisons[i]), dice_combinaison, 0]])

                remaining_moves = self.n_identical_dice[i] - \
                    self.hash.reverse_hash_function(dice_combinaison)

                for _ in range(5-remaining_moves.sum()):
                    remaining_moves[np.random.randint(
                        0, self.dice_max_value)] += 1

                self.dice[i] = 0

                k = 0
                for j in range(self.dice_max_value):
                    for _ in range(remaining_moves[j]):
                        self.dice[i, k*self.dice_max_value+j] = 1
                        k += 1

            # Second dice roll

            self.update_one_hot_dice()

            states = np.concatenate([self.is_box_checked, self.value_box /
                                    self.normalization_boxes_reward, self.n_identical_dice_one_hot], axis=1)

            available_combinaisons = self.hash.hash_function(
                self.n_identical_dice)

            outputs = self.model_dice_2(
                [states, available_combinaisons])[0].numpy()
            outputs_reweighted = outputs/outputs.sum(axis=1, keepdims=True)

            for i in range(self.n_games):
                if mode == "test":
                    dice_combinaison = np.random.choice(
                        np.arange(462), p=outputs_reweighted[i])
                else:
                    dice_combinaison = np.argmax(outputs_reweighted[i])
                self.history_model_dice_2[i] += copy(
                    [[(states[i], available_combinaisons[i]), dice_combinaison, 0]])

                remaining_moves = self.n_identical_dice[i] - \
                    self.hash.reverse_hash_function(dice_combinaison)

                for _ in range(5-remaining_moves.sum()):
                    remaining_moves[np.random.randint(
                        0, self.dice_max_value)] += 1

                self.dice[i] = 0

                k = 0
                for j in range(self.dice_max_value):
                    for _ in range(remaining_moves[j]):
                        self.dice[i, k*self.dice_max_value+j] = 1
                        k += 1

            # Box to check

            self.available_moves()
            states = [np.concatenate([self.is_box_checked, self.value_box/self.normalization_boxes_reward,
                                     self.n_identical_dice_one_hot, self.available_boxes], axis=1), self.available_boxes.astype(np.float32)]

            outputs = self.model_box(states)[0].numpy()

            outputs_reweighted = outputs/outputs.sum(axis=1, keepdims=True)
            self.decision = np.zeros(self.n_games, dtype=np.int32)

            for i in range(self.n_games):
                if mode == "test":
                    self.decision[i] = np.random.choice(
                        np.arange(self.n_boxes), p=outputs_reweighted[i])
                else:
                    self.decision[i] = np.argmax(outputs_reweighted[i])
            self.reward = np.zeros(self.n_games, dtype=np.float32)
            self.determine_intermediate_reward(turn)

            for i in range(self.n_games):
                self.history_model_box[i] += copy(
                    [[(states[0][i], states[1][i]), self.decision[i], self.reward[i]]])

        for i in range(self.n_games):
            for j in range(self.n_boxes):
                sparse_reward_dice_1 = (self.history_model_box[i][self.n_boxes - j - 1][2] if j == 0 else self.history_model_box[i]
                                        [self.n_boxes - j - 1][2] + self.gamma*self.history_model_dice_1[i][self.n_boxes - j][2])
                sparse_reward_dice_2 = (self.history_model_box[i][self.n_boxes - j - 1][2] if j == 0 else self.history_model_box[i]
                                        [self.n_boxes - j - 1][2] + self.gamma*self.history_model_dice_2[i][self.n_boxes - j][2])
                sparse_reward_box = (self.history_model_box[i][self.n_boxes - j - 1][2] if j == 0 else self.history_model_box[i]
                                     [self.n_boxes - j - 1][2] + self.gamma*self.history_model_box[i][self.n_boxes - j][2])
                self.history_model_dice_1[i][self.n_boxes -
                                             j-1][2] = copy(sparse_reward_dice_1)
                self.history_model_dice_2[i][self.n_boxes -
                                             j-1][2] = copy(sparse_reward_dice_2)
                self.history_model_box[i][self.n_boxes -
                                          j-1][2] = copy(sparse_reward_box)

    # Inference mode

    def initialize_inference(self, path):

        class Unit():
            def __init__(unit, hidden, hidden_value, hidden_policy, output_policy_dice, output_policy_box):
                unit.hidden = hidden
                unit.hidden_value = hidden_value
                unit.hidden_policy = hidden_policy
                unit.output_policy_dice = output_policy_dice
                unit.output_policy_box = output_policy_box

        unit = Unit(1024, 512, 512, 462, 13)

        # model dice 1
        self.model_dice_1 = RLModel(unit, mode="dice")
        self.model_dice_1([tensorflow.random.normal(
            (1, 63), 0, 1), tensorflow.ones((1, 462))])
        self.model_dice_1.load_weights(f"{path}/model_dice_1.h5")

        # model dice 2
        self.model_dice_2 = RLModel(unit, mode="dice")
        self.model_dice_2([tensorflow.random.normal(
            (1, 63), 0, 1), tensorflow.ones((1, 462))])
        self.model_dice_2.load_weights(f"{path}/model_dice_2.h5")

        # model box
        self.model_box = RLModel(unit, mode="box")
        self.model_box([tensorflow.random.normal(
            (1, 76), 0, 1), tensorflow.ones((1, 13))])
        self.model_box.load_weights(f"{path}/model_box.h5")

        self.initialize()

    def show_dice(self, step):
        """
        Display the dice values for the bot
        """
        if step == 1:
            roll_dice = np.random.randint(
                0, self.dice_max_value, (self.n_games, self.n_dice))
            self.update_dice(roll_dice)
            self.update_one_hot_dice()

        elif step == 2 or step == 3:
            pass

        else:
            raise Exception(f"Step {step} is not an acceptable value (should be between 1 and 3)")

        # print("\nRolling dice...")
        # for i in range(self.n_dice):
            # print(
            #     f"dice {i+1} has value {1+self.dice[0, i*self.dice_max_value:(i+1)*self.dice_max_value].argmax()}")

    def displays_dice_rerolled(self, step, silence=False):

        states = np.concatenate([self.is_box_checked, self.value_box /
                                self.normalization_boxes_reward, self.n_identical_dice_one_hot], axis=1)
        available_combinaisons = self.hash.hash_function(self.n_identical_dice)

        if step == 1:
            outputs = self.model_dice_1([states, available_combinaisons])
        elif step == 2:
            outputs = self.model_dice_2([states, available_combinaisons])
        else:
            raise Exception(f"Step {step} is not an acceptable value (should be between 1 and 2)")
            
        # print(f"The bot predicts that it will get a discounted final score of {(outputs[1].numpy()[0, 0]).astype(np.int32)}")

        outputs_reweighted = outputs[0].numpy(
        )/outputs[0].numpy().sum(axis=1, keepdims=True)
        
        n_dice_to_reroll = self.hash.reverse_hash_function(
            outputs_reweighted[0].argmax())

        dice_to_reroll = []

        for i in range(self.n_dice):
            x = np.argmax(
                self.dice[0, self.dice_max_value*i:self.dice_max_value*(i+1)])
            if n_dice_to_reroll[x] > 0:
                n_dice_to_reroll[x] -= 1
                dice_to_reroll.append(i)

        if (n_dice_to_reroll != 0).any():
            raise Exception("Combination of dice played not accepted")

        dice_to_reroll = np.array(dice_to_reroll)

        if not silence:
            if dice_to_reroll.size == 0:
                print("The bot does not roll any dice")
            elif dice_to_reroll.size < self.n_dice:
                dice_string = ", ".join([str(dice+1) for dice in dice_to_reroll])
                print(f"The bot throws the dice {dice_string}")
            elif dice_to_reroll.size == self.n_dice:
                print("The bot re-rolls all the dice")
            else:
                raise Exception(
                    f"The array dice_to_reroll : {dice_to_reroll} does not have a correct size")

        for i in dice_to_reroll:
            self.dice[0, self.dice_max_value*i:self.dice_max_value*(i+1)] = 0
            self.dice[0, self.dice_max_value*i+np.random.randint(0, 6)] = 1

        self.update_one_hot_dice()

        return dice_to_reroll

    def show_scoreboard_points(self, round, silence):

        self.available_moves()

        states = [np.concatenate([self.is_box_checked, self.value_box/self.normalization_boxes_reward,
                                  self.n_identical_dice_one_hot, self.available_boxes], axis=1), self.available_boxes.astype(np.float32)]
        outputs = self.model_box(states)

        # print(f"The bot predicts that it will get a discounted final score of {(outputs[1].numpy()[0, 0]).astype(np.int32)}")

        outputs_reweighted = outputs[0].numpy(
        )/outputs[0].numpy().sum(axis=1, keepdims=True)

        self.decision = np.zeros(self.n_games, dtype=np.int32)

        self.decision[0] = outputs_reweighted[0].argmax().astype(np.int32)

        self.reward = np.zeros(self.n_games, dtype=np.float32)
        self.determine_intermediate_reward(round)

        all_decisions = np.array(["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
                                 "Three of a kind", "Four of a kind", "Full House", "Small Straight", "Large Straight", "Chance", "Yahtzee"])

        if not silence:
            print(f"The bot decided to play the {all_decisions[self.decision[0]]}")

            print("\nSCOREBOARD")
            print("==================================")
            for i in range(self.n_boxes-2):
                if self.is_box_checked[0, i] == 1:
                    print(
                        f"{i+1} {all_decisions[i]}| {(self.value_box[0, i]*self.is_box_checked[0, i]).astype(np.int32)} points")
                else:
                    print(f"{i+1} {all_decisions[i]}|")
            if self.is_box_checked[0, 12] == 1:
                print(
                    f"12 {all_decisions[12]}| {(self.value_box[0, 12]*self.is_box_checked[0, 12]+self.value_box[0, 12]).astype(np.int32)} points")
            else:
                print(f"12 {all_decisions[12]}|")
            if self.is_box_checked[0, 11] == 1:
                print(
                    f"12 {all_decisions[11]}| {(self.value_box[0, 11]*self.is_box_checked[0, 11]).astype(np.int32)} points")
            else:
                print(f"11 {all_decisions[12]}|")
            print("==================================")

    def determine_move(self, dice_state, remaining_rules, history_rules, turn, silence=True):
        self.dice = np.zeros((1, self.dice_max_value*self.n_dice))
        for i in range(self.n_dice):
            self.dice[0, int(dice_state[i]+self.dice_max_value*i)] = 1
        self.update_one_hot_dice()
        self.is_box_checked[0] = remaining_rules
        self.value_box[0] = history_rules

        if turn == 1:
            dice_to_reroll = self.displays_dice_rerolled(1, silence)
            return dice_to_reroll
        elif turn == 2:
            dice_to_reroll = self.displays_dice_rerolled(2, silence)
            return dice_to_reroll
        elif turn == 3:
            self.show_scoreboard_points(self.n_boxes, silence)
            return self.decision[0]
        else:
            raise Exception(f"Turn {turn} is not an acceptable value (should be between 1 and 3)")
