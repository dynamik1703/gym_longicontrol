from gym.envs.registration import register

register(id='DeterministicTrack-v0',
         entry_point='gym_longicontrol.envs:DeterministicTrack',
         max_episode_steps=1800,
         kwargs={
             'car_id': 'BMW_electric_i3_2014',
             'speed_limit_positions': [0.0, 0.25, 0.5, 0.75],
             'speed_limits': [50, 80, 40, 50],
             'reward_weights': [1.0, 0.5, 1.0, 1.0],
             'energy_factor': 1.0
         })

register(id='StochasticTrack-v0',
         entry_point='gym_longicontrol.envs:StochasticTrack',
         max_episode_steps=1800,
         kwargs={
             'car_id': 'BMW_electric_i3_2014',
             'reward_weights': [1.0, 0.5, 1.0, 1.0],
             'energy_factor': 1.0
         })
