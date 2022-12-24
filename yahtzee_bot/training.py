import numpy as np
import tensorflow

from yahtzee_bot.data import CollectSampleExperiments
from yahtzee_bot.network import RLModel
from yahtzee_bot.ppo_loss import PPO


class Train():
    def __init__(self, args):
        self.n_ite = args.n_ite
        self.n_games = args.n_games
        self.hidden = args.hidden
        self.hidden_value = args.hidden_value
        self.hidden_policy = args.hidden_policy
        self.output_policy_dice = args.output_policy_dice
        self.output_policy_box = args.output_policy_box
        self.old_weights_path = args.old_weights_path
        self.save_weights_path = args.save_weights_path
        self.use_old_weights = args.use_old_weights
        self.do_save_weights = args.do_save_weights
        self.n_dice = args.n_dice
        self.max_value_dice = args.max_value_dice
        self.n_boxes = args.n_boxes
        self.normalization_boxes_reward = args.normalization_boxes_reward
        self.gamma = args.gamma
        self.max_iter = args.max_iter
        self.a1 = args.a1
        self.a2 = args.a2
        self.eps = args.eps
        self.lr = args.lr
        self.batch_size_dice = args.batch_size_dice
        self.batch_size_box = args.batch_size_box
        self.n_epochs = args.n_epochs
        self.freq = args.freq
        self.clipvalue = args.clipvalue

    def initialize(self):
        self.reward_history = np.zeros(self.n_ite, dtype=np.float32)

        self.best_model_dice_1 = RLModel(self, mode="dice")
        self.best_model_dice_2 = RLModel(self, mode="dice")
        self.best_model_box = RLModel(self, mode="box")

        model_dice_1 = RLModel(self, mode="dice")
        model_dice_2 = RLModel(self, mode="dice")
        model_box = RLModel(self, mode="box")

        self.collect_sample = CollectSampleExperiments(self.n_games, model_dice_1, model_dice_2, model_box, self.n_dice, self.max_value_dice,
                                                       self.n_boxes, self.normalization_boxes_reward, self.gamma)

    def update_network(self, from_best_to_current=True):
        if from_best_to_current:
            weights_model_dice_1 = np.array(
                self.best_model_dice_1.get_weights(), dtype=object)
            self.collect_sample.model_dice_1.set_weights(weights_model_dice_1)
            weights_model_dice_2 = np.array(
                self.best_model_dice_2.get_weights(), dtype=object)
            self.collect_sample.model_dice_2.set_weights(weights_model_dice_2)
            weights_model_box = np.array(
                self.best_model_box.get_weights(), dtype=object)
            self.collect_sample.model_box.set_weights(weights_model_box)
        else:
            weights_model_dice_1 = np.array(
                self.collect_sample.model_dice_1.get_weights(), dtype=object)
            self.best_model_dice_1.set_weights(weights_model_dice_1)
            weights_model_dice_2 = np.array(
                self.collect_sample.model_dice_2.get_weights(), dtype=object)
            self.best_model_dice_2.set_weights(weights_model_dice_2)
            weights_model_box = np.array(
                self.collect_sample.model_box.get_weights(), dtype=object)
            self.best_model_box.set_weights(weights_model_box)

    def save_current_network(self):
        self.collect_sample.model_dice_1.save_weights(
            f"{self.save_weights_path}/model_dice_1.h5")
        self.collect_sample.model_dice_2.save_weights(
            f"{self.save_weights_path}/model_dice_2.h5")
        self.collect_sample.model_box.save_weights(
            f"{self.save_weights_path}/model_box.h5")

    def train(self):
        self.initialize()

        self.collect_sample.model_dice_1([tensorflow.random.normal(
            (1, 63), 0, 1), tensorflow.ones((1, self.output_policy_dice))])
        self.collect_sample.model_dice_2([tensorflow.random.normal(
            (1, 63), 0, 1), tensorflow.ones((1, self.output_policy_dice))])
        self.collect_sample.model_box([tensorflow.random.normal(
            (1, 76), 0, 1), tensorflow.ones((1, self.output_policy_box))])

        self.best_model_dice_1([tensorflow.random.normal(
            (1, 63), 0, 1), tensorflow.ones((1, self.output_policy_dice))])
        self.best_model_dice_2([tensorflow.random.normal(
            (1, 63), 0, 1), tensorflow.ones((1, self.output_policy_dice))])
        self.best_model_box([tensorflow.random.normal(
            (1, 76), 0, 1), tensorflow.ones((1, self.output_policy_box))])

        if self.use_old_weights:
            self.collect_sample.model_dice_1.load_weights(
                f"{self.old_weights_path}/model_dice_1.h5")
            self.collect_sample.model_dice_2.load_weights(
                f"{self.old_weights_path}/model_dice_2.h5")
            self.collect_sample.model_box.load_weights(
                f"{self.old_weights_path}/model_box.h5")

        self.update_network(from_best_to_current=False)

        n_ite_not_improve = 0

        for ite in range(self.n_ite):

            if ite > self.max_iter and self.reward_history.max() > self.reward_history[ite-n_ite_not_improve-1:ite].max():
                n_ite_not_improve += 1

            if ite > self.max_iter and self.reward_history.max() > self.reward_history[ite-5:ite].max() and n_ite_not_improve == self.max_iter:
                print("Take older version")
                self.update_network(from_best_to_current=True)
                n_ite_not_improve = 0
            
            self.collect_sample.generate_sample()

            print(
                f"Reward statistics{self.collect_sample.average_reward_statistic()} at iteration {ite}")

            self.reward_history[ite] = self.collect_sample.average_reward_statistic()[
                1]

            if ite > 1 and self.reward_history[ite] >= self.reward_history[:ite].max():
                print("Update version")
                self.update_network(from_best_to_current=False)
                n_ite_not_improve = 0
                if self.do_save_weights:
                    self.save_current_network()

            # Model Dice 1

            optimization_parameters = self.a1, self.a2, self.eps*(1-ite/self.n_ite), self.lr*(
                1-ite/self.n_ite), self.batch_size_dice, self.n_epochs, self.freq, self.clipvalue

            states = np.concatenate([np.stack([self.collect_sample.history_model_dice_1[i][j][0][0] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0).astype(np.float32)
            one_D_actions = np.concatenate([np.stack([self.collect_sample.history_model_dice_1[i][j][1] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0)
            available_actions = np.concatenate([np.stack([self.collect_sample.history_model_dice_1[i][j][0][1] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0).astype(np.float32)
            actions = np.zeros((one_D_actions.size, 462), dtype=np.float32)
            actions[np.arange(one_D_actions.size), one_D_actions] = 1
            values = np.expand_dims(np.concatenate([np.stack([self.collect_sample.history_model_dice_1[i][j][2] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0), axis=1).astype(np.float32)

            PPO(self.collect_sample.model_dice_1, states,
                actions, available_actions, values, optimization_parameters)

            # Model Dice 2

            optimization_parameters = self.a1, self.a2, self.eps*(1-ite/self.n_ite), self.lr*(
                1-ite/self.n_ite), self.batch_size_dice, self.n_epochs, self.freq, self.clipvalue

            states = np.concatenate([np.stack([self.collect_sample.history_model_dice_2[i][j][0][0] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0).astype(np.float32)
            one_D_actions = np.concatenate([np.stack([self.collect_sample.history_model_dice_2[i][j][1] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0)
            available_actions = np.concatenate([np.stack([self.collect_sample.history_model_dice_2[i][j][0][1] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0).astype(np.float32)
            actions = np.zeros((one_D_actions.size, 462), dtype=np.float32)
            actions[np.arange(one_D_actions.size), one_D_actions] = 1
            values = np.expand_dims(np.concatenate([np.stack([self.collect_sample.history_model_dice_2[i][j][2] for i in range(
                self.collect_sample.n_games)], axis=0) for j in range(self.collect_sample.n_boxes)], axis=0), axis=1).astype(np.float32)

            PPO(self.collect_sample.model_dice_2, states,
                actions, available_actions, values, optimization_parameters)

            # Model Box

            optimization_parameters = self.a1, self.a2, self.eps*(1-ite/self.n_ite), self.lr*(
                1-ite/self.n_ite), self.batch_size_box, self.n_epochs, self.freq, self.clipvalue

            random_values = np.random.randint(
                0, self.collect_sample.n_boxes, self.collect_sample.n_games)
            states = np.stack([self.collect_sample.history_model_box[i][random_values[i]][0][0] for i in range(
                self.collect_sample.n_games)], axis=0).astype(np.float32)
            one_D_actions = np.stack([self.collect_sample.history_model_box[i][random_values[i]][1]
                                     for i in range(self.collect_sample.n_games)], axis=0)
            actions = np.zeros(
                (one_D_actions.size, self.collect_sample.n_boxes), dtype=np.float32)
            actions[np.arange(one_D_actions.size), one_D_actions] = 1
            available_actions = np.stack([self.collect_sample.history_model_box[i][random_values[i]][0][1] for i in range(
                self.collect_sample.n_games)], axis=0).astype(np.float32)
            values = np.expand_dims(np.stack([self.collect_sample.history_model_box[i][random_values[i]][2] for i in range(
                self.collect_sample.n_games)], axis=0), axis=1).astype(np.float32)

            PPO(self.collect_sample.model_box, states, actions,
                available_actions, values, optimization_parameters)

        self.collect_sample.generate_sample()
        print(
            f"reward_statistics{self.collect_sample.average_reward_statistic()} at iteration {ite+1}")
        self.reward_history[ite] = self.collect_sample.average_reward_statistic()[
            1]

        if ite > 1 and self.reward_history[ite] >= self.reward_history[:ite].max():
            print("update version")
            self.update_network(from_best_to_current=False)
            if self.do_save_weights:
                self.save_current_network()

        self.update_network(from_best_to_current=True)

        self.collect_sample.generate_sample(mode="inference")
        print(
            f"Final_reward_statistics{self.collect_sample.average_reward_statistic()} in inference mode")
        if self.do_save_weights:
            self.save_current_network()
