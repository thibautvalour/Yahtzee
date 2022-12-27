import numpy as np
import tensorflow as tf
import unittest

from . import training
from . import network
from . import data

Train = training.Train
RLModel = network.RLModel
CollectSampleExperiments = data.CollectSampleExperiments


class Args():
    def __init__(self, n_ite, n_games, hidden, hidden_value, hidden_policy, output_policy_dice, output_policy_box, old_weights_path, save_weights_path, use_old_weights, do_save_weights, n_dice, max_value_dice, n_boxes, normalization_boxes_reward, gamma, max_iter, a1, a2, eps, lr, batch_size_dice, batch_size_box, n_epochs, freq, clipvalue):
        self.n_ite = n_ite
        self.n_games = n_games
        self.hidden = hidden
        self.hidden_value = hidden_value
        self.hidden_policy = hidden_policy
        self.output_policy_dice = output_policy_dice
        self.output_policy_box = output_policy_box
        self.old_weights_path = old_weights_path
        self.save_weights_path = save_weights_path
        self.use_old_weights = use_old_weights
        self.do_save_weights = do_save_weights
        self.n_dice = n_dice
        self.max_value_dice = max_value_dice
        self.n_boxes = n_boxes
        self.normalization_boxes_reward = normalization_boxes_reward
        self.gamma = gamma
        self.max_iter = max_iter
        self.a1 = a1
        self.a2 = a2
        self.eps = eps
        self.lr = lr
        self.batch_size_dice = batch_size_dice
        self.batch_size_box = batch_size_box
        self.n_epochs = n_epochs
        self.freq = freq
        self.clipvalue = clipvalue


class TestTrain(unittest.TestCase):
    def test_initialize(self):
        args = Args(4, 128, 40, 50, 50, 462, 13, "", "", False, False, 5,
                    6, 13, 50., 0.9, 5, 0.5, 0.01, 0.2, 0.00025, 256, 1024, 5, 1., 1.)
        train = Train(args)
        train.initialize()
        expected = np.zeros(4, dtype=np.float32)
        self.assertTrue((train.reward_history == expected).all())

        self.assertTrue(isinstance(train.best_model_dice_1, RLModel))
        self.assertTrue(isinstance(train.best_model_dice_2, RLModel))
        self.assertTrue(isinstance(train.best_model_box, RLModel))

        self.assertTrue(isinstance(
            train.collect_sample, CollectSampleExperiments))

    def test_train(self):
        # Just test if the code does not crash
        args = Args(4, 128, 40, 50, 50, 462, 13, "", "", False, False, 5,
                    6, 13, 50., 0.9, 5, 0.5, 0.01, 0.2, 0.00025, 256, 1024, 5, 1., 1.)
        train = Train(args)
        train.train()
        self.assertEqual(train.reward_history.size, 4)
