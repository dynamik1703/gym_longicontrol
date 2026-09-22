import os
import pickle
import numpy as np

G = 9.81  # gravity
FR = 0.015  # friction
RHO = 1.2  # density air, kg/m³
M_S_MPH = 0.44704  # 1 mph = 0.44704 m/s
KM_H_M_S = 3.6  # 1 m/s = 3.6 km/h


class Cars:
    def __init__(self, car_id='BMW_electric_i3_2014'):
        """
        Build car object. Get power estimation and car specs.
        The training of the power estimator is done in a preprocessing step.
        :param car_id: 'MB_electric_Bclass_2015',
                       'VW_electric_Golf_2015',
                       'BMW_electric_i3_2014'
        """
        self.specs = get_specs(car_id)
        self.power_model = get_power_estimator(car_id)

        self.action_scaling_vecs()

    def action_scaling_vecs(self):
        """
        Calculate the vectors to rescale an action [-1,1] to a corresponding
        velocity-dependent acceleration.
        Limit power and acceleration to stay within a data-rich region
        of the v-a-map --> important for a valid power-prediction
        """
        vel_vec = np.arange(1, self.specs['velocity_limits'][1] + 1, 1)

        acc_pos_vec = self.calc_acceleration_from_power(
            vel_vec, self.specs['power_limits'][1])
        acc_neg_vec = self.calc_acceleration_from_power(
            vel_vec, self.specs['power_limits'][0])
        acc_0_vec = self.calc_acceleration_from_power(vel_vec, 0)

        acc_pos_vec = np.min([
            acc_pos_vec,
            np.ones(len(acc_pos_vec)) * self.specs['acceleration_limits'][1]
        ],
                             axis=0)
        acc_neg_vec = np.max([
            acc_neg_vec,
            np.ones(len(acc_neg_vec)) * self.specs['acceleration_limits'][0]
        ],
                             axis=0)

        # TODO: Find better solution :)
        # This is kind of a workaround. Roman got the values for 0 from the
        # data, which seems difficult to implement here. So the added 1.0 in
        # acc_pos_vec is handcrafted.
        self.vel_vec = np.append(0, vel_vec)
        self.acc_pos_vec = np.append(1.0, acc_pos_vec)
        self.acc_neg_vec = np.append(0.0, acc_neg_vec)
        self.acc_0_vec = np.append(0.0, acc_0_vec)

    def calc_acceleration_from_power(self, velocity, power):
        """
        Physically calculates the corresonding acceleration for a specific
        power at a specific velocity.
        :param vel: velocity
        :param P: power
        :return: acceleration
        """
        # TODO: make formular readable - comment!
        acceleration = (power / (velocity * self.specs['mass']) * 1000 - (
            self.specs['cW'] * self.specs['frontal_area'] * RHO * velocity**2 /
            (2 * self.specs['mass']) + FR * G))
        return acceleration

    def get_acceleration_from_action(self, velocity, action):
        """
        Calculates the acceleration corresponding to the chosen action.
        :param velocity: velocity
        :param action: action [-1,1]
        :return acceleration: the action rescaled to an velocity-dependent
                              acceleration
        """
        action_min = np.interp(velocity,
                               self.vel_vec,
                               self.acc_neg_vec,
                               left=0,
                               right=-1e-6)
        action_0 = np.interp(velocity,
                             self.vel_vec,
                             self.acc_0_vec,
                             left=0,
                             right=-1e-6)
        action_max = np.interp(velocity,
                               self.vel_vec,
                               self.acc_pos_vec,
                               left=0.6258544444444445,
                               right=-1e-6)

        action_lim = action_max if action > 0 else action_min
        acceleration = (action_lim - action_0) * abs(action) + action_0
        return acceleration

    def get_predicted_power(self, velocity, acceleration):
        """
        Uses a trained regression model (e.g. neural net) to predict the power.
        :param velocity: velocity in m/s
        :param acceleration: accceleration in m/s²
        :return power: electric power in kW
        """
        if isinstance(velocity, np.ndarray):
            X = np.stack((velocity, acceleration), axis=-1)
            return self.power_model.predict(X)
        else:
            X = np.array([velocity, acceleration]).reshape(1, -1)
            return self.power_model.predict(X)[0]


def get_power_estimator(car_id):
    """
    If not already done, build power estimator - otherwise just load it.
    :param car_id: 'MB_electric_Bclass_2015', 'VW_electric_Golf_2015',
                   'BMW_electric_i3_2014'
    :return: power estimator (sklearn regressor)
    """
    model_fname = os.path.join(os.path.dirname(__file__),
                               'assets/vehicle/' + car_id + '.pkl')
    if not os.path.exists(model_fname):
        create_power_estimator(car_id)
    with open(model_fname, 'rb') as f:
        model = pickle.load(f)
    return model


def create_power_estimator(car_id):
    # TODO: download d3 data from their webpage and omit local files
    # create pandas df from d3 data
    try:
        import pandas as pd
    except ImportError:
        raise ImportError('''
        Cannot import pandas.
        HINT: you can install pandas directly via 'pip install pandas'.
        ''')
    rel_dir = os.path.join(os.path.dirname(__file__),
                           'assets/vehicle/d3/' + car_id)
    for file in os.listdir(rel_dir):
        if 'Test Data.txt' in file:
            if '~lock.' in file:
                raise Exception(f'close {file.split("lock.")[1]}!')
            fname = os.path.join(rel_dir, file)
            data = pd.read_csv(fname, sep='\t', engine='python')
            print(fname)
            try:
                df = df.append(data, ignore_index=True)
            except NameError:
                df = data
    dt = round(abs(df['Time[sec]'][0] - df['Time[sec]'][1]), 4)
    df['velocity[m_s]'] = df['Dyno_Speed[mph]'] * M_S_MPH
    df['acceleration[m_s2]'] = np.ediff1d(df['velocity[m_s]'], to_end=0) / dt
    df['power[kW]'] = df['HV_Battery_Voltage[V]'] \
        * df['HV_Battery_Current[A]'] / 1000
    df = df.dropna(subset=('velocity[m_s]', 'acceleration[m_s2]', 'power[kW]'))

    df.to_csv('tests.csv', sep='\t')
    X = df[['velocity[m_s]', 'acceleration[m_s2]']]
    y = df['power[kW]']

    from sklearn.neural_network import MLPRegressor
    MLP = MLPRegressor(hidden_layer_sizes=(10, 10, 10),
                       learning_rate_init=1e-4,
                       random_state=0,
                       max_iter=200,
                       solver='adam',
                       batch_size=128,
                       verbose=True)
    model = MLP.fit(X, y)

    model_fname = os.path.join(os.path.dirname(__file__),
                               'assets/vehicle/' + car_id + '.pkl')
    with open(model_fname, 'wb') as f:
        pickle.dump(model, f)


def get_specs(car_id):
    """
    get car specific values
    :param car_id: 'MB_electric_Bclass_2015',
                   'VW_electric_Golf_2015',
                   'BMW_electric_i3_2014'
    """
    # TODO: add more cars!
    if car_id == 'BMW_electric_i3_2014':
        return {
            'mass': 1443,
            'frontal_area': 2.38,
            'cW': 0.29,
            'acceleration_limits': [-3, 3],
            'velocity_limits': [0, 37],
            'power_limits': [-50, 75]
        }
    else:
        raise NotImplementedError('So far only MB_electric_Bclass_2015,\
                                  VW_electric_Golf_2015, BMW_electric_i3_2014 \
                                  are implemented.')


if __name__ == '__main__':
    car_id = 'BMW_electric_i3_2014'
    model = get_power_estimator(car_id)
    # create_power_estimator(car_id)
