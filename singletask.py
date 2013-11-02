import numpy as np
from gridworld import *

class LinearGaussianRewardModel(object):
    def __init__(self, num_colors):
        pass


class SingleTaskBayesianAgent(Agent):
    """
    A Bayesian RL agent that views all the domains as drawn from the same distribution.
    """
    def __init__(self, domains):
        Agent.__init__(self, domains)

    def episode_starting(self, idx, state):
        pass

    def episode_over(self, idx):
        pass

    def get_action(self, idx):
        pass

    def set_state(self, idx, state):
        Agent.set_state(self, idx, state)

    def observe_reward(self, idx, r):
        pass
