# gym-longicontrol
We aim to combine real-world motivated RL with easy accessibility within a highly relevant problem: the stochastic longitudinal control of an autonomous vehicle.
The LongiControl environment consists of a data based electric vehicle model (the Downloadable Dynamometer Database [D3](https://www.anl.gov/es/downloadable-dynamometer-database) of the Argonne National Laboratory is used here) and a single-lane track with stochastic speed restrictions. The state of the agent includes the actual speed, previous acceleration, current speed limit and at most the next two speed limits as long as they are within a visual range of 150m. The agent selects the acceleration of the vehicle and receives as a reward a combination of speed, energy consumption, jerk and a measure for speeding. 
LongiControl could be used to elaborate various challenges within Reinforcement Learning. E.g. MORL due to excplicitly contradictory reward terms (minimize energy consumption, travel time, jerk) or SafeRL (comply with speed limits).


Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_longicontrol,
  author = {Dohmen, Jan T. and Liessner, Roman and Friebel, Christoph and Bäker, Bernard},
  title = {LongiControl Environment for OpenAI Gym},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dynamik1703/gym-longicontrol}},
}

and
@misc{gym_longicontrol_abstract,
  author = {Liessner, Roman and Dohmen, Jan T. and Friebel, Christoph and Bäker, Bernard},
  title = {LongiControl: A New Reinforcement Learning Environment},
  year = {2019},
  publisher = {5th World Congress on Electrical Engineering and Computer Systems and Science (EECSS'19)},
  doi = {10.11159/cist19.118},
}

```


## Requirements
- Python 3
- OpenAI Gym
- Numpy, Scikit-learn, Pandas

- Pytorch to train/use the given RL agent
- Pyglet, Matplotlib, Seaborn for visualization


## Install
```
cd gym-longicontrol
pip install -e .
```


## Usage
Instances of the environment can be created and handled similar to other Gym environments:
- `gym.make('gym_longicontrol:DeterministicTrack-v0')`
- `gym.make('gym_longicontrol:StochasticTrack-v0')`


## Example with built-in RL agent

### Training
```
cd /gym-longicontrol/rl/pytorch
python main.py --save_id 99
```

A trained agent is given in `/gym-longicontrol/rl/pytorch/out`:
- .tar ... model and weights
- .out ... quick overview of the training results
- .npy ... more detailed information about the course of training (can be used within the jupyter notebook `/gym_longicontrol/rl/pytorch/monitor.ipynb`)


### Visualize
Load the trained model an visualize an example track:
```
cd /gym-longicontrol/rl/pytorch
python main.py --load_id 9 --env_id StochasticTrack-v0 -vis
```

It is also possible to save it as mp4 video:
```
cd /gym-longicontrol/rl/pytorch
python main.py --load_id 9 --env_id DeterministicTrack-v0 -vis -rec
```

<p align="center">
<img src="/img/trained_agent.gif" width=600 height=270>
</p>


