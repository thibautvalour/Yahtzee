import numpy as np

from data import CollectSampleExperiments
from network import Unit, DiceReRoll, BoxChoice


def test_initialize():
    unit_dice_1 = Unit(64, 32, 32, 32)
    unit_dice_2 = Unit(64, 32, 32, 32)
    unit_box = Unit(64, 32, 32, 32)
    model_dice_1 = DiceReRoll(unit_dice_1)
    model_dice_2 = DiceReRoll(unit_dice_2)
    model_box = BoxChoice(unit_box)
    collect_sample = CollectSampleExperiments(
        10, 5, 6, 13, 50., 200., model_dice_1, model_dice_2, model_box)

    collect_sample.initialize()

    assert (collect_sample.value_box == np.zeros((10, 14), dtype=np.float32)
            ).all(), "The array value_box has not the correct value"
    assert (collect_sample.is_box_checked == np.zeros((10, 13), dtype=np.int32)).all(
    ), "The array is_box_checked has not the correct value"
    assert (collect_sample.dice == np.zeros((10, 30), dtype=np.int32)
            ).all(), "The array dice has not the correct value"
    assert collect_sample.history_model_dice_1 == [[] for _ in range(
        10)], "The list history_model_dice_1 has not the correct value"
    assert collect_sample.history_model_dice_2 == [[] for _ in range(
        10)], "The list history_model_dice_2 has not the correct value"
    assert collect_sample.history_model_box == [[] for _ in range(
        10)], "The list history_model_box has not the correct value"


def test_available_moves():
    unit_dice_1 = Unit(64, 32, 32, 32)
    unit_dice_2 = Unit(64, 32, 32, 32)
    unit_box = Unit(64, 32, 32, 32)
    model_dice_1 = DiceReRoll(unit_dice_1)
    model_dice_2 = DiceReRoll(unit_dice_2)
    model_box = BoxChoice(unit_box)
    collect_sample = CollectSampleExperiments(
        7, 5, 6, 13, 50., 200., model_dice_1, model_dice_2, model_box)

    collect_sample.initialize()

    collect_sample.dice = np.array([
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

    collect_sample.value_box = np.array([
        [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 16, 0, 0,  0, 0, 0, 25, 0, 0, 0, 0],
        [3, 0, 0, 12, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0],
        [3, 6, 9, 12, 15, 12,  0, 8, 25, 0, 0, 25, 50, 0],
        [2, 0, 6, 0, 20, 12,  0, 0, 0, 0, 0, 27, 0, 0],
        [6, 4, 15, 12, 10, 24,  25, 0, 0, 0, 0, 18, 50, 100],
        [0, 2, 0, 4, 5, 6,  20, 20, 25, 30, 40, 18, 50, 0]
    ], dtype=np.float32)

    collect_sample.is_box_checked = np.array([
        [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0,  0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1,  1, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 1,  0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1,  1, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.int32)

    collect_sample.available_moves()

    is_any_box_reached = np.array([1, 1, 1, 0, 1, 1, 0], dtype=np.int32)

    available_boxes = np.array([
        [0, 1, 0, 1, 1, 0,  0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1,  0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1,  1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0,  0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)

    assert (collect_sample.is_any_box_reached == is_any_box_reached).all(
    ), "The array is_any_box_reached has not the correct value"
    assert (collect_sample.available_boxes == available_boxes).all(
    ), "The array available_boxes has not the correct value"
