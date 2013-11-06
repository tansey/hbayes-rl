import numpy as np
from gridworld import *
from mdp_solver import *

class Observation(object):
    def __init__(self, episode, state, reward, goal = False):
        self.episode = episode
        self.state = state
        self.reward = reward
        self.goal = goal

class LinearGaussianRewardModel(object):
    def __init__(self, num_colors):
        self.observations = []

    def add_reward_observation(self, observation):
        self.observations.append(observation)
        self.calculate_posterior()

    def calculate_posterior(self):
        pass

    def predict_value(self, state):
        pass


class SingleTaskBayesianAgent(Agent):
    """
    A Bayesian RL agent that views all the domains as drawn from the same distribution.
    """
    def __init__(self, width, height, num_colors, num_domains, name=None):
        super(SingleTaskBayesianAgent, self).__init__(width, height, num_colors, num_domains, name)
        self.model = LinearGaussianRewardModel(num_colors)
        self.value_function = np.array((num_domains, width, height))
        self.state_vector = np.array((num_domains, width, height, num_colors * NUM_RELATIVE_CELLS))

    def episode_starting(self, idx, state):
        super(SingleTaskBayesianAgent, self).episode_starting(idx, state)

    def episode_over(self, idx):
        super(SingleTaskBayesianAgent, self).episode_over(idx)

    def get_action(self, idx):
        pass

    def set_state(self, idx, state):
        super(SingleTaskBayesianAgent, self).set_state(idx, state)

    def observe_reward(self, idx, r):
        super(SingleTaskBayesianAgent, self).observe_reward(idx, r)
        self.model.add_reward_observation(Observation(idx, ))
