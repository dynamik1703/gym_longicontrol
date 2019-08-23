import numpy as np
import os

from gym_longicontrol.envs import car

import gym
from gym import spaces
from gym.utils import seeding


class DeterministicTrack(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 car_id='BMW_electric_i3_2014',
                 speed_limit_positions=[0.0, 0.25, 0.5, 0.75],
                 speed_limits=[50, 80, 40, 50],
                 reward_weights=[1.0, 0.5, 1.0, 1.0],
                 energy_factor=1.0):
        """
        The car is longitudinal guided with uniformly accelerated motion.
        Basic track with defined speed limits, speed limit positions and length.
        :param car_id: string car id
        :param speed_limit_positions: list speed limit positions in km
        :param speed_limits: list speed limits in km/h
        :param reward_weights: list reward weights (forward, energy, jerk, shock)
        :param energy_factor: float additional energy weight factor
        """
        self.car = car.Cars(car_id=car_id)
        self.speed_limit_positions = speed_limit_positions
        self.speed_limits = speed_limits
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

    def step(self, action):
        """
        Take one 10Hz step:
        Update time, position, velocity, jerk, limits.
        Check if episode is done.
        Get reward.
        :param action: float within (-1, 1)
        :return: state, reward, done, info
        """
        action = np.clip(action, -1, 1)
        assert self.action_space.contains(action),\
            f'{action} ({type(action)}) invalid shape or bounds'
        self.acceleration = self.car.get_acceleration_from_action(
            self.velocity, action)[0]
        # s = 0.5 * a * t² + v0 * t + s0
        self.position += (0.5 * self.acceleration * self.dt**2 +
                          self.velocity * self.dt)
        # v = a * t + v0
        self.velocity += self.acceleration * self.dt

        (self.current_speed_limit, self.future_speed_limits,
         self.future_speed_limit_distances) = self.sensor(self.position)

        self.time += self.dt
        self.jerk = abs((self.acceleration - self.prev_acceleration)) / self.dt
        self.prev_acceleration = self.acceleration

        self.done = bool(self.position >= self.track_length)
        reward_list = self.get_reward()
        info = {
            'position': self.position,
            'velocity': self.velocity * 3.6,
            'acceleration': self.acceleration,
            'jerk': self.jerk,
            'time': self.time,
            'energy': self.total_energy_kWh
        }
        state = self.feature_scaling(self.get_state())
        reward = np.array(reward_list).dot(np.array(self.reward_weights))
        return state, reward, self.done, info

    def reset(self):
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

    def get_reward(self):
        """
        Calculate the reward for this time step.
        Requires current limits, velocity, acceleration, jerk, time.
        Get predicted energy rate (power) from car data.
        Use negative energy as reward.
        Use negative jerk as reward (scaled).
        Use velocity as reward (scaled).
        Use a shock penalty as reward.
        :return: reward
        """
        # calc forward or velocity reward
        reward_forward = abs(self.velocity - self.current_speed_limit)
        forward_max = self.current_speed_limit
        reward_forward /= forward_max

        # calc energy reward
        power_kW = self.car.get_predicted_power(self.velocity,
                                                self.acceleration)
        energy_kWh = power_kW * self.dt / 3600
        reward_energy = energy_kWh
        energy_max = self.car.specs['power_limits'][1] * self.dt / 3600
        reward_energy /= energy_max
        self.total_energy_kWh += energy_kWh

        # calc jerk reward
        reward_jerk = self.jerk
        jerk_max = np.diff(self.car.specs['acceleration_limits'])[0] / self.dt
        reward_jerk /= jerk_max

        # calc shock reward
        reward_shock = 1 if self.velocity > self.current_speed_limit else 0

        reward_list = [
            -reward_forward, -reward_energy, -reward_jerk, -reward_shock
        ]
        return reward_list

    def get_state(self):
        """
        Wrapper to update state with current feature values.
        :return: state
        """
        return np.hstack(
            (self.velocity, self.prev_acceleration, self.current_speed_limit,
             self.future_speed_limits, self.future_speed_limit_distances,
             self.energy_factor))

    def sensor(self, position):
        current_speed_limit = 0.0
        current_speed_limit_i = 0
        next_speed_limit = 0.0
        next_speed_limit_distance = 0.0
        next2_speed_limit = 0.0
        next2_speed_limit_distance = 0.0
        for i, (pos, sl) in enumerate(
                zip(self.speed_limit_positions, self.speed_limits)):
            if pos <= position:
                current_speed_limit = sl
                current_speed_limit_i = i
        if current_speed_limit_i + 1 > len(self.speed_limits) - 1:
            next_speed_limit = current_speed_limit
            next_speed_limit_distance = self.sensor_range
        elif (self.speed_limit_positions[current_speed_limit_i + 1] - position
              > self.sensor_range):
            next_speed_limit = current_speed_limit
            next_speed_limit_distance = self.sensor_range
        else:
            next_speed_limit = self.speed_limits[current_speed_limit_i + 1]
            next_speed_limit_distance = self.speed_limit_positions[
                current_speed_limit_i + 1] - position

        if current_speed_limit_i + 2 > len(self.speed_limits) - 1:
            next2_speed_limit = next_speed_limit
            next2_speed_limit_distance = self.sensor_range
        elif (self.speed_limit_positions[current_speed_limit_i + 2] - position
              > self.sensor_range):
            next2_speed_limit = next_speed_limit
            next2_speed_limit_distance = self.sensor_range
        else:
            next2_speed_limit = self.speed_limits[current_speed_limit_i + 2]
            next2_speed_limit_distance = self.speed_limit_positions[
                current_speed_limit_i + 2] - position
        future_speed_limits = [next_speed_limit, next2_speed_limit]
        future_speed_limit_distances = [
            next_speed_limit_distance, next2_speed_limit_distance
        ]
        return (current_speed_limit, future_speed_limits,
                future_speed_limit_distances)

    def manage_speed_limits(self):
        """
        Prepare speed limits and corresponding positions.
        A limit for -inf is needed due to backwards driving.
        """
        self.speed_limits = np.array(self.speed_limits,
                                     dtype=np.float32) / 3.6  # in m/s
        self.speed_limit_positions = np.array(self.speed_limit_positions,
                                              dtype=np.float32) * 1000  # in m
        assert 0.0 in self.speed_limit_positions, \
            'speed_limit_positions does not contain 0.0. ' \
            'Also add an initial speed limit.'
        assert len(self.speed_limits) == len(self.speed_limit_positions), \
            'speed_limit_positions doesnt have the same size as speed_limits.'

        self.speed_limit_positions = np.insert(self.speed_limit_positions, 0,
                                               -np.inf)
        self.speed_limits = np.insert(self.speed_limits, 0,
                                      self.speed_limits[0])

    def feature_scaling(self, state):
        """
        Min-Max-Scaler: scale X' = (X-Xmin) / (Xmax-Xmin)
        :param state:
        :return: scaled state
        """
        return (state - self.state_min) / (self.state_max - self.state_min)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        screen_width = 1000
        screen_height = 450
        clearance_x = 80
        clearance_y = 10
        zero_x = 0.25 * screen_width
        visible_track_length = 1000
        scale_x = screen_width / visible_track_length

        if self.viewer is None:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from gym_longicontrol.envs import rendering
            self.viewer = rendering.Viewer(width=screen_width,
                                           height=screen_height)

            rel_dir = os.path.join(os.path.dirname(__file__),
                                   'assets/track/img')

            # start and finish line
            fname = os.path.join(rel_dir, 'start_finish_30x100.png')
            start = rendering.Image(fname,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            start.position = (zero_x, clearance_y)
            self.viewer.components['start'] = start
            finish = rendering.Image(fname,
                                     rel_anchor_y=0,
                                     batch=self.viewer.batch,
                                     group=self.viewer.background)
            finish.position = (zero_x + scale_x * self.track_length,
                               clearance_y)
            self.viewer.components['finish'] = finish

            self.viewer.components['signs'] = []

            # speedometer
            fname = os.path.join(rel_dir, 'speedometer_232x190.png')
            speedometer = rendering.Image(fname,
                                          rel_anchor_y=0,
                                          batch=self.viewer.batch,
                                          group=self.viewer.background)
            speedometer.position = (screen_width - 110 - clearance_x,
                                    220 + clearance_y)
            self.viewer.components['speedometer'] = speedometer

            fname = os.path.join(rel_dir, 'needle_6x60.png')
            needle = rendering.Image(fname,
                                     rel_anchor_y=0.99,
                                     batch=self.viewer.batch,
                                     group=self.viewer.foreground)
            needle.position = (screen_width - 110 - clearance_x,
                               308 + clearance_y)
            self.viewer.components['needle'] = needle

            fname = os.path.join(rel_dir, 'needle_6x30.png')
            needle_sl = rendering.Image(fname,
                                        rel_anchor_y=2.6,
                                        batch=self.viewer.batch,
                                        group=self.viewer.background)
            needle_sl.position = (screen_width - 110 - clearance_x,
                                  308 + clearance_y)
            self.viewer.components['needle_sl'] = needle_sl

            # info label
            velocity0_label = rendering.Label(text='km/h',
                                              batch=self.viewer.batch,
                                              group=self.viewer.foreground,
                                              anchor_x='left',
                                              font_size=12,
                                              color=(255, 255, 255, 255))
            velocity0_label.position = (screen_width - 110 - clearance_x,
                                        267 + clearance_y)
            self.viewer.components['velocity0_label'] = velocity0_label
            energy0_label = rendering.Label(text='kWh',
                                            batch=self.viewer.batch,
                                            group=self.viewer.foreground,
                                            anchor_x='left',
                                            font_size=12,
                                            color=(255, 255, 255, 255))
            energy0_label.position = (screen_width - 155 - clearance_x,
                                      238 + clearance_y)
            self.viewer.components['energy0_label'] = energy0_label
            time0_label = rendering.Label(text='min',
                                          batch=self.viewer.batch,
                                          group=self.viewer.foreground,
                                          anchor_x='left',
                                          font_size=12,
                                          color=(255, 255, 255, 255))
            time0_label.position = (screen_width - 49 - clearance_x,
                                    238 + clearance_y)
            self.viewer.components['time0_label'] = time0_label

            velocity_label = rendering.Label(text=str(int(self.velocity *
                                                          3.6)),
                                             batch=self.viewer.batch,
                                             group=self.viewer.foreground,
                                             anchor_x='left',
                                             font_size=12,
                                             color=(255, 255, 255, 255))
            velocity_label.position = (screen_width - 150 - clearance_x,
                                       267 + clearance_y)
            self.viewer.components['velocity_label'] = velocity_label
            energy_label = rendering.Label(text=str(
                round(self.total_energy_kWh, 2)),
                                           batch=self.viewer.batch,
                                           group=self.viewer.foreground,
                                           anchor_x='left',
                                           font_size=12,
                                           color=(255, 255, 255, 255))
            energy_label.position = (screen_width - 200 - clearance_x,
                                     238 + clearance_y)
            self.viewer.components['energy_label'] = energy_label
            m, s = divmod(self.time, 60)
            time_label = rendering.Label(text=f'{m:02.0f}:{s:02.0f}',
                                         batch=self.viewer.batch,
                                         group=self.viewer.foreground,
                                         anchor_x='left',
                                         font_size=12,
                                         color=(255, 255, 255, 255))
            time_label.position = (screen_width - 99 - clearance_x,
                                   238 + clearance_y)
            self.viewer.components['time_label'] = time_label

            # info figures
            self.viewer.history['velocity'] = []
            self.viewer.history['speed_limit'] = []
            self.viewer.history['position'] = []
            self.viewer.history['acceleration'] = []
            sns.set_style('whitegrid')
            self.fig = plt.Figure((640 / 80, 200 / 80), dpi=80)
            info = rendering.Figure(self.fig,
                                    rel_anchor_x=0,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            info.position = (clearance_x - 40, 225 + clearance_y)
            self.viewer.components['info'] = info

            # car
            fname = os.path.join(rel_dir, 'car_80x40.png')
            car = rendering.Image(fname,
                                  rel_anchor_x=1,
                                  batch=self.viewer.batch,
                                  group=self.viewer.foreground)
            car.position = (zero_x, 50 + clearance_y)
            self.viewer.components['car'] = car

        # speed limit signs
        if not self.viewer.components['signs']:
            from gym_longicontrol.envs import rendering
            rel_dir = os.path.join(os.path.dirname(__file__),
                                   'assets/track/img')
            for sl, slp in zip(self.speed_limits[1:],
                               self.speed_limit_positions[1:]):
                fname = os.path.join(rel_dir,
                                     f'sign_60x94_{str(int(sl * 3.6))}.png')
                sign = rendering.Image(fname,
                                       rel_anchor_y=0,
                                       batch=self.viewer.batch,
                                       group=self.viewer.background)
                sign.position = (zero_x + scale_x * (slp - self.position),
                                 100 + clearance_y)
                self.viewer.components['signs'].append(sign)

        # updates
        for sl, slp, sign in zip(self.speed_limits[1:],
                                 self.speed_limit_positions[1:],
                                 self.viewer.components['signs']):
            distance = slp - self.position
            if distance >= self.sensor_range:
                sign.opacity = 64
            else:
                sign.opacity = 255
            if -zero_x - 50 <= scale_x * distance <= screen_width - zero_x + 50:
                sign.position = (zero_x + scale_x * distance,
                                 100 + clearance_y)
                sign.visible = True
            else:
                sign.visible = False
        self.viewer.components['start'].position = (zero_x + scale_x *
                                                    (0 - self.position),
                                                    clearance_y)
        self.viewer.components['finish'].position = (
            zero_x + scale_x * (self.track_length - self.position),
            clearance_y)

        # car turns red if speed limit is exceeded
        if self.velocity > self.current_speed_limit:
            self.viewer.components['car'].color = (128, 0, 0)
        else:
            self.viewer.components['car'].color = (255, 255, 255)

        # speedometer
        deg = 60.0 + self.current_speed_limit * 3.6 * 1.5  # 1km/h =^ 1.5°
        self.viewer.components['needle_sl'].rotation = deg
        deg = 60.0 + self.velocity * 3.6 * 1.5  # 1km/h =^ 1.5°
        self.viewer.components['needle'].rotation = deg
        self.viewer.components['velocity_label'].text = str(
            int(self.velocity * 3.6))
        self.viewer.components['energy_label'].text = str(
            round(self.total_energy_kWh, 2))
        m, s = divmod(self.time, 60)
        self.viewer.components['time_label'].text = f'{m:02.0f}:{s:02.0f}'

        # info figures
        self.viewer.history['velocity'].append(self.velocity * 3.6)
        self.viewer.history['speed_limit'].append(self.current_speed_limit *
                                                  3.6)
        self.viewer.history['position'].append(self.position)
        self.viewer.history['acceleration'].append(self.acceleration)
        self.viewer.components['info'].visible = False
        if self.viewer.plot_fig or mode == 'rgb_array':
            self.viewer.components['info'].visible = True
            self.fig.clf()
            ax = self.fig.add_subplot(121)
            ax.plot(self.viewer.history['position'],
                    self.viewer.history['velocity'],
                    lw=2,
                    color='k')
            ax.plot(self.viewer.history['position'],
                    self.viewer.history['speed_limit'],
                    lw=1.5,
                    ls='--',
                    color='r')
            ax.set_xlabel('Position in m')
            ax.set_ylabel('Velocity in km/h')
            ax.set_xlim(
                (0.0, max(500, self.position + (500 - self.position) % 500)))
            ax.set_ylim((0.0, 130))

            ax2 = self.fig.add_subplot(122)
            ax2.plot(self.viewer.history['position'],
                     self.viewer.history['acceleration'],
                     lw=2,
                     color='k')
            ax2.set_xlabel('Position in m')
            ax2.set_ylabel('Acceleration in m/s²')
            ax2.set_xlim(
                (0.0, max(500, self.position + (500 - self.position) % 500)))
            ax2.set_ylim((-5.0, 5.0))

            self.fig.tight_layout()
            self.viewer.components['info'].figure = self.fig

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def reset_viewer(self):
        if self.viewer is not None:
            for key in self.viewer.history:
                self.viewer.history[key] = []
            self.viewer.components['signs'] = []

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
