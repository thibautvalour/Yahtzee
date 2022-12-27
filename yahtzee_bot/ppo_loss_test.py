from yahtzee_bot.ppo_loss import compute_gradients, PPO
import numpy as np
import tensorflow as tf
import unittest
from yahtzee_bot.network import RLModel
from yahtzee_bot.network_test import Unit


class TestPPOLoss(unittest.TestCase):
    def test_compute_gradients(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        unit = Unit(512, 512, 512, 462, 13)
        model = RLModel(unit, "dice")
        states = tf.random.normal([10, 462], 0, 1)
        available_actions = tf.ones((10, 462))
        actions = np.zeros((10, 462), dtype=np.float32)
        random_actions = np.random.randint(0, 462, (10))
        actions[np.arange(random_actions.size), random_actions] = 1
        fixed_policy = tf.random.normal([10, 462], 0.5, 0.1)
        advantages = tf.random.normal([10], 0, 1)
        values = tf.random.normal([10], 10, 1)
        a1 = 0.5
        a2 = 0.01
        eps = 0.2
        compute_gradients(optimizer, model, states, available_actions,
                          actions, fixed_policy, advantages, values, a1, a2, eps)

    def test_PPO(self):
        unit = Unit(512, 512, 512, 462, 13)
        model = RLModel(unit, "dice")
        states = np.random.normal(0, 1, (100, 462))
        actions = np.zeros((100, 462), dtype=np.float32)
        random_actions = np.random.randint(0, 462, (100))
        actions[np.arange(random_actions.size), random_actions] = 1
        available_actions = np.ones((100, 462))
        values = np.random.normal(10, 1, (100))
        a1 = 0.5
        a2 = 0.01
        eps = 0.2
        lr = 0.00025
        batch_size = 4
        epochs = 5
        freq = 1.
        clipvalue = 1.
        optimization_parameters = a1, a2, eps, lr, batch_size, epochs, freq, clipvalue
        PPO(model, states, actions, available_actions,
            values, optimization_parameters)
