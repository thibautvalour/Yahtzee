import numpy as np
import tensorflow as tf


def compute_gradients(optimizer, model, states, available_actions, actions, fixed_policy, advantages, values, a1, a2, eps):

    with tf.GradientTape() as tape:

        if len(available_actions) != 0:
            expected_policy, expected_values = model(
                [states, available_actions])
        else:
            expected_policy, expected_values = model(states)
        rt = tf.reduce_sum(actions*expected_policy /
                           (fixed_policy+1e-9), axis=1, keepdims=True)
        clip_loss = tf.reduce_mean(tf.minimum(tf.multiply(rt, advantages), tf.multiply(
            tf.clip_by_value(rt, 1-eps, 1+eps), advantages)), axis=1, keepdims=True)
        value_loss = tf.square(expected_values-values)
        entropy_loss = tf.reduce_sum(-tf.multiply(tf.math.log(
            expected_policy+1e-9), expected_policy), axis=1, keepdims=True)
        total_loss = -tf.reduce_mean(clip_loss-a1*value_loss+a2*entropy_loss)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def PPO(model, states, actions, available_actions, values, optimization_parameters):
    a1, a2, eps, lr, batch_size, epochs, freq, clipvalue = optimization_parameters

    n = states.shape[0]

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=clipvalue)

    for _ in range(epochs):

        random_shuffle = tf.range(n)
        tf.random.shuffle(random_shuffle)

        states = states[random_shuffle]
        actions = actions[random_shuffle]
        if len(available_actions) != 0:
            available_actions = available_actions[random_shuffle]
        values = values[random_shuffle]
        for i in range(int(freq*(n//batch_size))):

            selected_moves = np.arange(batch_size*i, batch_size*(i+1))
            semi_states = states[selected_moves]
            if len(available_actions) != 0:
                semi_available_actions = available_actions[selected_moves]
            else:
                semi_available_actions = []
            semi_actions = actions[selected_moves]
            semi_values = values[selected_moves]

            if len(available_actions) != 0:
                semi_fixed_policy, semi_fixed_values = model(
                    [semi_states, semi_available_actions])
            else:
                semi_fixed_policy, semi_fixed_values = model(semi_states)

            semi_advantages = semi_values-semi_fixed_values

            compute_gradients(optimizer, model, semi_states, semi_available_actions, semi_actions,
                              semi_fixed_policy, semi_advantages, semi_values, a1, a2, eps)
