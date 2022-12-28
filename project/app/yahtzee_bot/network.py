from numpy import inf
import tensorflow
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Multiply
from keras.layers import Add
from keras.activations import softmax
from keras.activations import relu


class RLModel(keras.Model):
    """
    Model that generates the best choice of the checkbox

    Parameters
    ----------
    units : unit class
        Number of neurons par layer

    Returns
    ----------
    model : keras model
        Neural network model used to find the best choice
    """

    def __init__(self, units, mode, **kwargs):
        super().__init__(**kwargs)
        self.hidden = Dense(units.hidden, activation="relu", name="hidden")
        self.hidden_value = Dense(
            units.hidden_value, activation="relu", name="hidden_value")
        self.hidden_policy = Dense(
            units.hidden_policy, activation="relu", name="hidden_policy")
        self.output_value = Dense(1, activation="linear", name="output_value")
        if mode == "dice":
            self.output_policy = Dense(
                units.output_policy_dice, activation="linear", name="output_policy")
        elif mode == "box":
            self.output_policy = Dense(
                units.output_policy_box, activation="linear", name="output_policy")
        else:
            raise Exception(f"Mode {mode} is not valid")

    def call(self, input):
        dice_box_input, mask_input = input

        hidden = self.hidden(dice_box_input)
        hidden_policy = self.hidden_policy(hidden)
        output_policy = self.output_policy(hidden_policy)
        inf_mask = -relu(inf*(1-2*mask_input))
        output_policy = Multiply()([output_policy, mask_input])
        output_policy = Add()([output_policy, inf_mask])
        output_policy = softmax(output_policy)

        hidden_value = self.hidden_value(hidden)
        output_value = self.output_value(hidden_value)

        return output_policy, output_value
