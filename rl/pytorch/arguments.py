import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='RL SAC LongiControl',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.format_help()

    # general things
    parser.add_argument('--visualize',
                        '-vis',
                        action='store_true',
                        help='Visualize an episode. Only possible if a \n'
                        'trained model is loaded via --load_id.\n'
                        'Keys:\n'
                        'ENTER: show/hide information figures \n'
                        'SPACE: stop/continue visualization \n'
                        '(default: False)')

    parser.add_argument('--record',
                        '-rec',
                        action='store_true',
                        help='Record each evaluation run and save it as \n'
                        '.mp4 in \out directory \n'
                        '(default: False)')

    parser.add_argument('--save_id',
                        metavar='',
                        type=int,
                        default=0,
                        help='save_id sets a number for naming the saved \n'
                        'model and history. \n'
                        '(default: 0)')

    parser.add_argument('--load_id',
                        metavar='',
                        type=int,
                        default=None,
                        help='If load_id is not None the specified model is \n'
                        'loaded and used for the simulation. \n'
                        '(default: None)')

    parser.add_argument('--seed',
                        metavar='',
                        type=int,
                        default=2,
                        help='Set random seed for torch and numpy (gym) \n'
                        '(default: 2)')

    # SAC specifics
    parser.add_argument('--replay_buffer_capacity',
                        '-buf',
                        metavar='',
                        type=int,
                        default=1e6,
                        help='Specify the max number of experience samples \n'
                        '(default: 1e6)')

    parser.add_argument('--num_epochs',
                        '-ep',
                        metavar='',
                        type=int,
                        default=1e4,
                        help='Number of training epochs. \n'
                        '(default: 1e4)')

    parser.add_argument('--num_steps_per_epoch',
                        '-steps',
                        metavar='',
                        type=int,
                        default=1e3,
                        help='Number of steps per epoch. \n'
                        '(default: 1e3)')

    parser.add_argument('--discount_factor_gamma',
                        '-gamma',
                        metavar='',
                        type=float,
                        default=0.99,
                        help='Typical RL discount factor gamma.\n'
                        'Typical Range: 0 - 1 (default: 0.99)')

    parser.add_argument('--soft_update_factor_tau',
                        '-tau',
                        metavar='',
                        type=float,
                        default=0.01,
                        help='Soft update factor for target value network \n'
                        '(default: 0.01)')

    parser.add_argument('--optimization_batch',
                        '-batch',
                        metavar='',
                        type=int,
                        default=256,
                        help='optimization_batch is the number \n'
                        'of experience samples used for one \n'
                        'gradient descent update. \n'
                        '(default: 256)')

    parser.add_argument('--adam_lr',
                        '-lr',
                        metavar='',
                        type=float,
                        default=0.001,
                        help='lr corresponds to the strength of each \n'
                        'gradient descent update step (Adam Optimizer). \n'
                        'This should typically be decreased if training \n'
                        'is unstable, and the reward does not consistently \n'
                        'increase. \n'
                        'Typical Range: 1e-5 - 1e-3 (default: 0.001)')

    parser.add_argument('--hidden_layer_sizes',
                        metavar='',
                        nargs='+',
                        type=int,
                        default=[64, 64],
                        help='Specify the neural net architecture.\n'
                        '(default: 64 64)')

    # environment specifics
    parser.add_argument('--car_id',
                        metavar='',
                        type=str,
                        default='BMW_electric_i3_2014',
                        help='car_id corresponds to the used car model. \n'
                        'There is only one car model available at the \n'
                        'moment. \n'
                        '(default: BMW_electric_i3_2014)')

    parser.add_argument('--env_id',
                        metavar='',
                        type=str,
                        default='DeterministicTrack-v0',
                        help='env_id corresponds to the used environment.\n'
                        'Choose either \n'
                        'DeterministicTrack-v0 or StochsticTrack-v0 \n'
                        '(default: DeterministicTrack-v0)')

    parser.add_argument('--reward_weights',
                        '-rw',
                        metavar='',
                        nargs='+',
                        type=float,
                        default=[1.0, 0.5, 1.0, 1.0],
                        help='reward_weights shape the reward. \n'
                        'The order to be used is: forward, energy, \n'
                        'jerk, shock \n'
                        '(default: 1.0 0.5 1.0 1.0)')

    parser.add_argument('--energy_factor',
                        metavar='',
                        type=float,
                        default=1.0,
                        help='energy_factor is multiplied with the energy. \n'
                        '(default: 1.0)')

    parser.add_argument('--speed_limit_positions',
                        metavar='',
                        nargs='+',
                        type=float,
                        default=[0.0, 0.25, 0.5, 0.75],
                        help='For DeterministicTrack speed_limit_positions \n'
                        'is the list of positions for which the speed \n'
                        'limits are defined. Entries are absolute values \n'
                        'in km and must correspond to track_length. \n'
                        'List has to start with position 0.0, otherwise \n'
                        'an error occurs! Lists for limits and positions \n'
                        'have to be of equal length.\n'
                        '(default: 0.0 0.25 0.5 0.75)')

    parser.add_argument('--speed_limits',
                        metavar='',
                        nargs='+',
                        type=int,
                        default=[50, 80, 40, 50],
                        help='For DeterministicTrack speed_limits is the \n'
                        'list of speed limits in km/h at the given \n'
                        'positions defined in speed_limit_positions. \n'
                        'Lists for limits and positions have to be of \n'
                        'equal length. \n'
                        '(default: 50 80 40 50)')

    args = parser.parse_args()

    return args
