import numpy as np
import unittest

from . import data
from . import network
from . import network_test
from . import hash

CollectSampleExperiments = data.CollectSampleExperiments
RLModel = network.RLModel
Unit = network_test.Unit
Hash = hash.Hash


class TestData(unittest.TestCase):
    def test_init(self):
        n_games = 10
        unit = Unit(512, 512, 512, 462, 13)
        model_dice_1 = RLModel(unit, "dice")
        model_dice_2 = RLModel(unit, "dice")
        model_box = RLModel(unit, "box")
        collect = CollectSampleExperiments(n_games, model_dice_1, model_dice_2, model_box)
        self.assertEqual(collect.n_dice, 5)
        self.assertEqual(collect.dice_max_value, 6)
        self.assertEqual(collect.normalization_boxes_reward, 50.)
        self.assertEqual(collect.gamma, 0.9)
        self.assertTrue(isinstance(collect.hash, Hash))

    def test_initialize(self):
        n_games = 10
        unit = Unit(512, 512, 512, 462, 13)
        model_dice_1 = RLModel(unit, "dice")
        model_dice_2 = RLModel(unit, "dice")
        model_box = RLModel(unit, "box")
        collect = CollectSampleExperiments(n_games, model_dice_1, model_dice_2, model_box)
        collect.initialize()
        expected_value_box = np.zeros((10, 14), dtype=np.float32)
        expected_is_box_checked = np.zeros((10, 13), dtype=np.float32)
        expected_dice = np.zeros((10, 30), dtype=np.float32)
        expected_history_model_dice_1 = [[] for _ in range(10)]
        expected_history_model_dice_2 = [[] for _ in range(10)]
        expected_history_model_box = [[] for _ in range(10)]

        self.assertTrue((collect.value_box == expected_value_box).all())
        self.assertTrue((collect.is_box_checked == expected_is_box_checked).all())
        self.assertTrue((collect.dice == expected_dice).all())
        self.assertEqual(collect.history_model_dice_1, expected_history_model_dice_1)
        self.assertEqual(collect.history_model_dice_2, expected_history_model_dice_2)
        self.assertEqual(collect.history_model_box, expected_history_model_box)

    def test_update_dice(self):
        n_games = 2
        collect = CollectSampleExperiments(n_games, None, None, None)
        dice = np.array([[4, 5, 0, 2, 1], [3, 2, 2, 3, 1]])
        expected = np.array([[0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 1,  1, 0, 0, 0, 0, 0,  0, 0, 1, 0, 0, 0,  0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 0,  0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0,  0, 1, 0, 0, 0, 0]])
        collect.update_dice(dice)
        self.assertTrue((collect.dice == expected).all())

    def test_update_one_hot_dice(self):
        n_games = 2
        collect = CollectSampleExperiments(n_games, None, None, None)
        collect.dice = np.array([[0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 1,  1, 0, 0, 0, 0, 0,  0, 0, 1, 0, 0, 0,  0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 0,  0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0,  0, 1, 0, 0, 0, 0]])
        collect.update_one_hot_dice()
        excepted_n_identical_dice = np.array([[1, 1, 1, 0, 1, 1], [0, 1, 2, 2, 0, 0]])
        excepted_n_identical_dice_one_hot = np.array([[0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  0, 0, 1, 0, 0, 0,  0, 0, 1, 0, 0, 0,  1, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0]])
        self.assertTrue((collect.n_identical_dice == excepted_n_identical_dice).all())
        self.assertTrue((collect.n_identical_dice_one_hot == excepted_n_identical_dice_one_hot).all())

    def test_available_moves(self):
        n_games = 7
        collect = CollectSampleExperiments(n_games, None, None, None)
        collect.initialize()
        collect.dice = np.array([
            [0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 1, 0,  0, 0, 0, 1,
                0, 0,  0, 1, 0, 0, 0, 0,  0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0,  0, 0, 0, 0,
                0, 1,  0, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1,  0, 0, 0, 1, 0, 0,  0, 0, 0, 0,
                0, 1,  0, 0, 0, 0, 0, 1,  0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0,  0, 1, 0, 0, 0, 0,  1, 0, 0, 0,
                0, 0,  0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  0, 1, 0, 0,
                0, 0,  0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 1, 0,  0, 0, 0, 0,
                1, 0,  0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 1,  0, 0, 0, 0,
                0, 1,  0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 1]
        ], dtype=np.int32)
        
        collect.value_box = np.array([
            [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 16, 0, 0,  0, 0, 0, 25, 0, 0, 0, 0],
            [3, 0, 0, 12, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0],
            [3, 6, 9, 12, 15, 12,  0, 8, 25, 0, 0, 25, 50, 0],
            [2, 0, 6, 0, 20, 12,  0, 0, 0, 0, 0, 27, 0, 0],
            [6, 4, 15, 12, 10, 24,  25, 0, 0, 0, 0, 18, 50, 100],
            [0, 2, 0, 4, 5, 6,  20, 20, 25, 30, 40, 18, 50, 0]
        ], dtype=np.float32)

        collect.is_box_checked = np.array([
            [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1,  1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1,  0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1,  1, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.int32)

        collect.available_moves()

        test_is_any_box_reached = np.array([1, 1, 1, 0, 1, 1, 0], dtype=np.int32)

        test_available_boxes = np.array([
            [0, 1, 0, 1, 1, 0,  0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 1,  0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1,  1, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,  0, 0, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int32)

        test_is_yahtzee_bonus = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int32)

        self.assertTrue((collect.is_any_box_reached == test_is_any_box_reached).all())
        self.assertTrue((collect.available_boxes == test_available_boxes).all())
        self.assertTrue((collect.is_yahtzee_bonus == test_is_yahtzee_bonus).all())

    def test_determine_intermediate_reward(self):
        n_games = 8
        collect = CollectSampleExperiments(n_games, None, None, None)
        collect.initialize()

        collect.reward = np.zeros(collect.n_games, dtype=np.float32)

        collect.decision = np.array([4, 11, 5, 9, 1, 10, 0, 2], dtype=np.int32)

        dice = np.array([
            [1, 3, 3, 4, 4],
            [0, 2, 3, 4, 5],
            [3, 3, 5, 5, 5],
            [0, 1, 1, 1, 4],
            [1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5],
            [2, 2, 2, 4, 5]
        ], dtype=np.int32)

        collect.update_dice(dice)
        collect.update_one_hot_dice()

        collect.value_box = np.array([
            [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 16, 0, 0,  0, 0, 0, 25, 0, 0, 0, 0],
            [3, 0, 0, 12, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0],
            [3, 6, 9, 12, 15, 12,  0, 8, 25, 0, 0, 25, 50, 0],
            [2, 0, 6, 0, 20, 12,  0, 0, 0, 0, 0, 27, 0, 0],
            [6, 4, 15, 12, 10, 0,  25, 0, 0, 0, 0, 18, 50, 100],
            [0, 2, 0, 4, 5, 6,  20, 20, 25, 30, 40, 18, 50, 0],
            [5, 10, 0, 20, 25, 30,  0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)

        collect.is_box_checked = np.array([
            [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1,  1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1,  0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0,  1, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int32)

        collect.is_any_box_reached = np.array(
            [1, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)

        collect.is_yahtzee_bonus = np.array(
            [0, 0, 0, 0, 0, 1, 1, 0], dtype=np.int32)

        collect.determine_intermediate_reward(12)

        test_value_box = np.array([
            [0, 0, 0, 0, 10, 0,  0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 16, 0, 0,  0, 0, 0, 25, 0, 19, 0, 0],
            [3, 0, 0, 12, 0, 18,  0, 0, 0, 0, 0, 0, 0, 0],
            [3, 6, 9, 12, 15, 12,  0, 8, 25, 0, 0, 25, 50, 0],
            [2, 10, 6, 0, 20, 12,  0, 0, 0, 0, 0, 27, 0, 0],
            [6, 4, 15, 12, 10, 0,  25, 0, 0, 0, 40, 18, 50, 200],
            [0, 2, 0, 4, 5, 6,  20, 20, 25, 30, 40, 18, 50, 100],
            [5, 10, 9, 20, 25, 30,  0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)

        test_is_box_checked = np.array([
            [0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0,  1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int32)

        test_reward = np.array([10, 19, 18, 0, 10, 140, 100, 44], dtype=np.float32)

        self.assertTrue((collect.reward == test_reward).all())
        self.assertTrue((collect.is_box_checked == test_is_box_checked).all())
        self.assertTrue((collect.value_box == test_value_box).all())

    def test_determine_final_reward(self):
        n_games = 3
        collect = CollectSampleExperiments(n_games, None, None, None)

        collect.value_box = np.array([
            [3, 8, 12, 12, 20, 18,  21, 0, 25, 0, 0, 18, 0, 0],
            [2, 10, 6, 0, 20, 12,  0, 0, 0, 0, 0, 27, 0, 0],
            [0, 2, 0, 4, 5, 6,  20, 20, 25, 30, 40, 18, 50, 100]
        ], dtype=np.float32)

        test_final_reward = np.array([172, 77, 320], dtype=np.float32)
        
        self.assertTrue((collect.determine_final_reward() == test_final_reward).all())
