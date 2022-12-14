from argparse import ArgumentParser

from yahtzee_bot.training import Train

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
