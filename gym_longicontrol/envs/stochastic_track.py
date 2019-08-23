import numpy as np

from gym_longicontrol.envs import car
from gym_longicontrol.envs.deterministic_track import DeterministicTrack

from gym import spaces


class StochasticTrack(DeterministicTrack):
    def __init__(self,
                 car_id='BMW_electric_i3_2014',
                 reward_weights=[1.0, 0.5, 1.0, 1.0],
                 energy_factor=1.0):
        """
        The car is longitudinal guided with uniformly accelerated motion.
        Stochastic track with semi-random speed limits & speed limit positions.
        :param car_id: string car id
        :param reward_weights: list reward weights (forward, energy, jerk, shock)
        :param energy_factor: float additional energy weight factor
        """
        self.car = car.Cars(car_id=car_id)
        self.reward_weights = reward_weights
        self.energy_factor = energy_factor

        self.seed()

        self.track_length = 1000.0
        self.dt = 0.1  # in s
        self.sensor_range = 150.0  # in m

        self.position = 0.0  # in m; track is 1-dim, only one coord is needed
        self.velocity = 0.0  # in m/s
        self.acceleration = 0.0  # in m/s**2
        self.jerk = 0.0  # in m/s**3
        self.prev_acceleration = 0.0  # in m/s**2
        self.time = 0.0  # in s
        self.total_energy_kWh = 0.0  # in Wh

        self.done = False

        self.speed_limit_positions = None
        self.speed_limits = None
        self.manage_speed_limits()

        (self.current_speed_limit, self.future_speed_limits,
         self.future_speed_limit_distances) = self.sensor(self.position)

        self.state_max = np.hstack(
            (self.car.specs['velocity_limits'][1],
             self.car.specs['acceleration_limits'][1],
             self.car.specs['velocity_limits'][1],
             self.car.specs['velocity_limits'][1] * np.ones(2),
             self.sensor_range * np.ones(2), 5.0))
        self.state_min = np.hstack(
            (self.car.specs['velocity_limits'][0],
             self.car.specs['acceleration_limits'][0],
             self.car.specs['velocity_limits'][0],
             self.car.specs['velocity_limits'][0] * np.ones(2), np.zeros(2),
             0.0))

        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(1, ),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=self.state_min,
                                            high=self.state_max,
                                            dtype=np.float32)

        self.viewer = None

    def reset(self):
        self.manage_speed_limits()

        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.jerk = 0.0
        self.prev_acceleration = 0.0
        self.total_energy_kWh = 0.0
        self.done = False
        self.time = 0.0
        self.reset_viewer()
        (self.current_speed_limit, self.future_speed_limits,
         self.future_speed_limit_distances) = self.sensor(self.position)
        state = self.feature_scaling(self.get_state())
        return state

    def manage_speed_limits(self):
        """
        Prepare speed limits and corresponding positions.
        A limit for -inf is needed due to backwards driving.
        A semi-random speed limit is possibly placed every 100m.
        Speed limits are allowed to be between 20 and 100 km/h,
        following speed limits deviate max 40km/h.

        TODO:
        Find better routine...
        Especially try to 'simulate' traffic.
        """

        amount_of_speed_limits = max(
            self.np_random.randint(self.track_length // 100.0), 1)
        possible_positions = [
            i for i in np.arange(100.0, self.track_length, 100.0)
        ]
        random_positions = self.np_random.choice(possible_positions,
                                                 amount_of_speed_limits,
                                                 replace=False)
        random_positions.sort()
        speed_limit_positions_init = np.insert(random_positions, 0, 0.0)

        speed_limits_init = []
        v = 40
        sl = 20.0 / 3.6
        for position in speed_limit_positions_init:
            sl_min = max(20, v - 40)
            sl_max = min(100, v + 40)
            sl_array = np.linspace(
                sl_min, sl_max, (sl_max - sl_min) // 10 + 1, dtype=int) / 3.6
            sl_array = np.setdiff1d(sl_array, sl)
            sl = self.np_random.choice(sl_array)
            v = int(sl * 3.6)
            speed_limits_init.append(sl)

        speed_limits_init = np.array(speed_limits_init)
        self.speed_limit_positions = np.insert(speed_limit_positions_init, 0,
                                               -np.inf)
        self.speed_limits = np.insert(speed_limits_init, 0,
                                      speed_limits_init[0])
