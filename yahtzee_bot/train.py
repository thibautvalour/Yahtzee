from argparse import ArgumentParser
import numpy as np
import tensorflow

from data import CollectSampleExperiments
from network import RLModel
from ppo_loss import PPO


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

        self.collect_sample = CollectSampleExperiments(self.n_games, self.n_dice, self.max_value_dice,
                                                       self.n_boxes, self.normalization_boxes_reward, self.gamma, model_dice_1, model_dice_2, model_box)

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


if __name__ == "__main__":
    # create the parser
    parser = ArgumentParser()

    # add the arguments
    parser.add_argument("-n_ite", "--n_ite", type=int, required=True,
                        help="The number of steps in reinforcement learning training. An example of a typical value is 100")

    parser.add_argument("-n_games", "--n_games", type=int, required=True,
                        help="The number of games performed in parallel at each iteration. An example of a typical value is 4096")

    parser.add_argument("-hidden", "--hidden", type=int, default=1024,
                        help="The number of neurons in the common hidden layer")

    parser.add_argument("-hidden_value", "--hidden_value", type=int, default=512,
                        help="The number of neurons in the second hidden layer of the function value")

    parser.add_argument("-hidden_policy", "--hidden_policy", type=int, default=512,
                        help="The number of neurons in the second hidden layer of the policy function")

    parser.add_argument("-output_policy_dice", "--output_policy_dice", type=int, default=462,
                        help="The number of neurons in the second output layer of the policy function of the dice model")

    parser.add_argument("-output_policy_box", "--output_policy_box", type=int, default=13,
                        help="The number of neurons in the second output layer of the policy function of the box model")

    parser.add_argument("-old_weights_path", "--old_weights_path", type=str, default="",
                        help="The path to an existing old model (only if use_old_weights is True)")

    parser.add_argument("-save_weights_path", "--save_weights_path", type=str,
                        help="The path where the model will be saved (only if do_save_weights is True)")

    parser.add_argument("-use_old_weights", action="store_true",
                        help="The program will use old weights")

    parser.add_argument("-do_save_weights", action="store_true",
                        help="The program will save the weights")

    parser.add_argument("-n_dice", "--n_dice", type=int, default=5,
                        help="The number of dice used in the game (normally 5)")

    parser.add_argument("-max_value_dice", "--max_value_dice", type=int,
                        default=6, help="The number of sides of the die (normally 6)")

    parser.add_argument("-n_boxes", "--n_boxes", type=int, default=13,
                        help="The number of boxes to check (normally 13)")

    parser.add_argument("-normalization_boxes_reward", "--normalization_boxes_reward",
                        type=float, default=50., help="Normalization constant for the reward")

    parser.add_argument("-gamma", "--gamma", type=float,
                        default=0.9, help="Discount factor")

    parser.add_argument("-max_iter", "--max_iter", type=int, default=5,
                        help="The total number of iterations before taking points from a previous best model during training")

    parser.add_argument("-a1", "--a1", type=float,
                        default=0.5, help="a1 value in PPO loss")

    parser.add_argument("-a2", "--a2", type=float,
                        default=0.01, help="a2 value in PPO loss")

    parser.add_argument("-eps", "--eps", type=float,
                        default=0.2, help="eps value in PPO loss")

    parser.add_argument("-lr", "--lr", type=float,
                        default=0.00025, help="Initial learning rate")

    parser.add_argument("-batch_size_dice", "--batch_size_dice", type=int,
                        default=128, help="The size of the packages for training dice models")

    parser.add_argument("-batch_size_box", "--batch_size_box", type=int,
                        default=32, help="The size of the packages for training box model")

    parser.add_argument("-n_epochs", "--n_epochs", type=int,
                        default=5, help="The number of epochs during training")

    parser.add_argument("-freq", "--freq", type=float, default=1.,
                        help="The proportion of actions that will be chosen to train the neural networks")

    parser.add_argument("-clipvalue", "--clipvalue", type=float,
                        default=1., help="Gradient clip value")

    # parse the arguments
    args = parser.parse_args()

    # create the parser
    model = Train(args)

    # train the model
    model.train()
