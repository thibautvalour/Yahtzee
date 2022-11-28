import numpy as np

#from data import CollectSampleExperiments
#from network import DiceReRoll, BoxChoice
#from ppo_loss import PPO

optimization_parameters = 0.1, 0.01, 0.2, 1e-5, 32, 3, 0.2, 1


def train(n_ite, n_games, optimization_parameters, unit_dice_1, unit_dice_2, unit_box):
    model_dice_1 = DiceReRoll(unit_dice_1)
    model_dice_2 = DiceReRoll(unit_dice_2)
    model_box = BoxChoice(unit_box)
    collect_sample = CollectSampleExperiments(
        n_games, 5, 6, 13, 50., 200., model_dice_1, model_dice_2, model_box)

    for _ in range(n_ite):
        collect_sample.generate_sample()
        print(f"reward_statistics{collect_sample.average_reward_statistic()}")

        # Model Dice 1

        states = np.concatenate([np.stack([collect_sample.history_model_dice_1[i][j][0] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0).astype(np.float32)

        one_D_actions = np.concatenate([np.stack([collect_sample.history_model_dice_1[i][j][1] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0)

        actions = np.zeros(
            (one_D_actions.size, 2**collect_sample.n_dices), dtype=np.float32)

        actions[np.arange(one_D_actions.size), one_D_actions] = 1

        values = np.expand_dims(np.concatenate([np.stack([collect_sample.history_model_dice_1[i][j][2] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0), axis=1).astype(np.float32)

        PPO(collect_sample.model_dice_1, states,
            actions, [], values, optimization_parameters)

        # Model Dice 2

        states = np.concatenate([np.stack([collect_sample.history_model_dice_2[i][j][0] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0).astype(np.float32)

        one_D_actions = np.concatenate([np.stack([collect_sample.history_model_dice_2[i][j][1] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0)

        actions = np.zeros(
            (one_D_actions.size, 2**collect_sample.n_dices), dtype=np.float32)

        actions[np.arange(one_D_actions.size), one_D_actions] = 1

        values = np.expand_dims(np.concatenate([np.stack([collect_sample.history_model_dice_2[i][j][2] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0), axis=1).astype(np.float32)

        PPO(collect_sample.model_dice_2, states,
            actions, [], values, optimization_parameters)

        # Model Box

        states = np.concatenate([np.stack([collect_sample.history_model_box[i][j][0][0] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0).astype(np.float32)

        one_D_actions = np.concatenate([np.stack([collect_sample.history_model_box[i][j][1] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0)

        actions = np.zeros(
            (one_D_actions.size, collect_sample.n_boxes), dtype=np.float32)

        actions[np.arange(one_D_actions.size), one_D_actions] = 1

        available_actions = np.concatenate([np.stack([collect_sample.history_model_box[i][j][0][1] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0).astype(np.float32)

        values = np.expand_dims(np.concatenate([np.stack([collect_sample.history_model_box[i][j][2] for i in range(
            collect_sample.n_games)], axis=0) for j in range(collect_sample.n_boxes)], axis=0), axis=1).astype(np.float32)

        PPO(collect_sample.model_box, states, actions,
            available_actions, values, optimization_parameters)

    return collect_sample
