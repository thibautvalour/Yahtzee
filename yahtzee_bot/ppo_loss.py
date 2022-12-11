import numpy as np
import tensorflow as tf
from tqdm import tqdm


def compute_gradients(optimizer, model, states, available_actions, actions, fixed_policy, advantages, values, a1, a2, eps):
    """
    This code is defining a function called compute_gradients that takes several arguments:

    optimizer: a TensorFlow optimizer object
    model: a TensorFlow model
    states: a tensor of states
    available_actions: a tensor of available actions for each state
    actions: a tensor of actions taken for each state
    fixed_policy: a tensor of fixed policies for each state
    advantages: a tensor of advantages for each state
    values: a tensor of values for each state
    a1: a scalar value
    a2: a scalar value
    eps: a scalar value
    The code then uses TensorFlow's GradientTape to compute the gradient of the total loss with respect to the model's trainable variables. The total loss is the sum of several components:

    The clip loss, which is the sum of the minimum of the product of the expected policy and the advantages and the product of the clipped expected policy and the advantages for each state
    The entropy loss, which is the sum of the negative log of the expected policy for each state
    The value loss, which is the square of the difference between the expected values and the actual values for each state
    The code then applies the gradients to the model's trainable variables using the optimizer.
    """

    with tf.GradientTape() as tape:

        expected_policy, expected_values = model(
            [states, available_actions])

        rt = tf.reduce_sum(actions*expected_policy /
                           (fixed_policy+1e-9), axis=1, keepdims=True)
        clip_loss = tf.reduce_mean(tf.minimum(tf.multiply(rt, advantages), tf.multiply(
            tf.clip_by_value(rt, 1-eps, 1+eps), advantages)), axis=1, keepdims=True)
        entropy_loss = tf.reduce_sum(-tf.multiply(tf.math.log(
            expected_policy+1e-9), expected_policy), axis=1, keepdims=True)
        value_loss = tf.square(expected_values-values)
        total_loss = -tf.reduce_mean(clip_loss-a1*value_loss+a2*entropy_loss)

        # print(clip_loss[0][0].numpy(), -value_loss[0][0].numpy()*a1, entropy_loss[0][0].numpy()*a2)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def PPO(model, states, actions, available_actions, values, optimization_parameters):
    """
    This code is defining a function called PPO that takes several arguments:

    model: a TensorFlow model
    states: a tensor of states
    actions: a tensor of actions taken for each state
    available_actions: a tensor of available actions for each state
    values: a tensor of values for each state
    optimization_parameters: a list of optimization parameters, including a1, a2, eps, lr, batch_size, epochs, freq, and clipvalue
    The code then shuffles the input tensors and divides them into batches of size batch_size. For each batch, it computes the fixed policy and fixed values using the model, and then uses these to compute the advantages. It then calls the compute_gradients function to compute the gradients and apply them to the model's trainable variables. This process is repeated for a specified number of epochs.
    """

    a1, a2, eps, lr, batch_size, epochs, freq, clipvalue = optimization_parameters

    n = states.shape[0]

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipvalue)

    for _ in tqdm(range(epochs)):

        random_shuffle = tf.range(n)
        tf.random.shuffle(random_shuffle)

        states = states[random_shuffle]
        actions = actions[random_shuffle]
        available_actions = available_actions[random_shuffle]
        values = values[random_shuffle]

        for i in range(int(freq*(n//batch_size))):

            selected_moves = np.arange(batch_size*i, batch_size*(i+1))
            semi_states = states[selected_moves]
            semi_available_actions = available_actions[selected_moves]
            semi_actions = actions[selected_moves]
            semi_values = values[selected_moves]

            semi_fixed_policy, semi_fixed_values = model(
                [semi_states, semi_available_actions])

            semi_advantages = semi_values-semi_fixed_values

            compute_gradients(optimizer, model, semi_states, semi_available_actions, semi_actions,
                              semi_fixed_policy, semi_advantages, semi_values, a1, a2, eps)
