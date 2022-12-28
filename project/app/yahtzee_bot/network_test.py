import numpy as np
import unittest
import tensorflow as tf

from network import RLModel


class Unit():
    def __init__(unit, hidden, hidden_value, hidden_policy, output_policy_dice, output_policy_box):
        unit.hidden = hidden
        unit.hidden_value = hidden_value
        unit.hidden_policy = hidden_policy
        unit.output_policy_dice = output_policy_dice
        unit.output_policy_box = output_policy_box


class TestRLModel(unittest.TestCase):
    def test_call_1(self):
        tf.random.set_seed(53)
        test_cases = [
            (Unit(100, 10, 5, 462, 13), "dice", [
             tf.random.normal((2, 45)), tf.ones((2, 462))], (2, 462)),
            (Unit(10, 11, 53, 462, 13), "box", [
             tf.random.normal((1, 501)), tf.ones((1, 13))], (1, 13)),
        ]
        for unit, mode, input_value, expected in test_cases:
            model = RLModel(unit, mode)
            result = model(input_value)
            self.assertEqual(result[0].numpy().shape, expected)

    def test_call_2(self):
        tf.random.set_seed(53)
        test_cases = [
            (Unit(4, 2, 5, 462, 13), "dice", [
             tf.random.normal((5, 45)), tf.ones((5, 462))], 1.),
            (Unit(10, 11, 42, 462, 13), "box", [
             tf.random.normal((3, 501)), tf.ones((3, 13))], 1.),
        ]
        for unit, mode, input_value, expected in test_cases:
            model = RLModel(unit, mode)
            result = model(input_value)
            self.assertTrue(np.allclose(result[0].numpy().sum(
                axis=1), expected, atol=1e-3, rtol=1e-5))
