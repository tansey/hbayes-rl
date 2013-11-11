"""
NOTE: This file is currently FUBAR. Do not look at this for anything meaningful!
"""
import numpy as np
from gridworld import *
from mdp_solver import *
from sample_utils import sample_niw

class Observation(object):
    def __init__(self, episode, state, reward, reward_stdev, goal = False):
        self.episode = episode
        self.state = state
        self.reward = reward
        self.reward_stdev = reward_stdev
        self.goal = goal

class LinearGaussianRewardModel(object):
    def __init__(self, num_colors, posterior_samples=3000, burn_in=500, thin=10, proposal_variance=0.1):
        self.observations = []
        self.posterior_samples = posterior_samples
        self.burn_in = burn_in
        self.thin = thin
        self.proposal_variance = proposal_variance
        self.weights_mean_size = num_colors * NUM_RELATIVE_CELLS
        self.weights_cov_size = (self.weights_mean_size, self.weights_mean_size)
        self.weights_size = self.weights_mean_size
        # The parameters of the Normal-Inverse-Wishart prior (mu, lambda, nu, psi)
        # I am trying to use very vague parameters here, to get close to an uninformed prior.
        self.hyper_parameters = (np.zeros(self.weights_size), 1., self.weights_size, np.identity(self.weights_size) * 3.)
        self.episodes = 0
        self.bayes_estimates = (np.zeros(self.weights_size), np.identity(self.weights_size) * 3, np.zeros(self.weights_size))

    def add_reward_observation(self, observation, update=True):
        self.observations.append(observation)
        if update:
            self.update_beliefs()

    def update_beliefs(self):
        self.samples = self.calculate_posterior()
        self.bayes_estimates = (np.mean(s, axis=0) for s in self.samples)

    def calculate_posterior(self):
        mean_samples = []
        cov_samples = []
        weight_samples = []
        mean = np.zeros(self.weights_mean_size)
        cov = np.identity(self.weights_mean_size)
        weights = np.random.multivariate_normal(mean, cov)
        for i in range(self.posterior_samples):
            print i
            mean,cov = self.gibbs_weights_mean_cov(weights)
            weights = self.mcmc_weights(weights, mean, cov)
            if i >= self.burn_in and i % self.thin == 0:
                mean_samples.append(mean)
                cov_samples.append(cov)
                weight_samples.append(weights)
                print 'Iteration {0}'.format(i)
                print 'Phi: {0}'.format(mean)
                print 'Sigma: {0}'.format(cov_samples)
                print 'Weights: {0}'.format(weights)
                print ''

        return (mean_samples, cov_samples, weight_samples)

    def gibbs_weights_mean_cov(self, weights):
        avg_weights = weights #np.mean(weights, axis=0)
        post_mean = (self.hyper_parameters[1] * self.hyper_parameters[0] + self.episodes * avg_weights) / (self.hyper_parameters[1] + self.episodes)
        post_lambda = self.hyper_parameters[1] + self.episodes
        post_nu = self.hyper_parameters[2] + self.episodes
        deviation = avg_weights - self.hyper_parameters[0]
        #post_psi = self.hyper_parameters[3] + np.sum([np.dot(w - avg_weights, np.transpose(w - avg_weights)) for w in weights]) \
        post_psi = self.hyper_parameters[3] \
                        + self.hyper_parameters[1] * self.episodes / (self.hyper_parameters[1] + self.episodes) \
                        * np.dot(deviation, np.transpose(deviation))
        (sample_mean, sample_cov) = sample_niw(post_mean, post_lambda, post_nu, np.linalg.inv(post_psi))
        return (sample_mean, sample_cov)

    def mcmc_weights(self, prev_weights, mean, cov):
        cur_weights = np.copy(prev_weights)
        cur_likelihood = self.weights_likelihood(cur_weights, mean, cov)
        for i in range(self.burn_in):
            proposed = self.weights_proposal(cur_weights)
            proposal_likelihood = self.weights_likelihood(proposed, mean, cov)
            u = random.random()
            if math.log(u) < (proposal_likelihood - cur_likelihood):
                cur_weights = proposed
                cur_likelihood = proposal_likelihood
        return cur_weights

    def weights_proposal(self, prev_weights):
        return np.random.multivariate_normal(prev_weights, np.identity(self.weights_size) * self.proposal_variance)

    def weights_likelihood(self, weights, mean, cov):
        sum_rewards = sum([(x.reward - np.dot(np.transpose(weights), x.state)) ** 2 / (x.reward_stdev ** 2) for x in self.observations])
        prior_term = np.dot(np.dot(np.transpose(weights - mean), np.linalg.inv(cov)), weights - mean)
        return -0.5 * (sum_rewards + prior_term) # log prob

    def predict_value(self, state):
        return np.dot(self.bayes_estimates[0], state)            


class SingleTaskBayesianAgent(Agent):
    """
    A Bayesian RL agent that views all the domains as drawn from the same distribution.
    """
    def __init__(self, width, height, num_colors, num_domains, name=None, steps_per_policy=1):
        super(SingleTaskBayesianAgent, self).__init__(width, height, num_colors, num_domains, name)
        self.model = LinearGaussianRewardModel(num_colors)
        self.value_function = np.array((num_domains, width, height))
        self.state_vector = np.array((num_domains, width, height, num_colors * NUM_RELATIVE_CELLS))
        self.steps_per_policy = steps_per_policy
        self.steps_since_update = 0

    def episode_starting(self, idx, state):
        super(SingleTaskBayesianAgent, self).episode_starting(idx, state)
        self.steps_since_update = 0
        mdp = self.sample_mdp()


    def episode_over(self, idx):
        super(SingleTaskBayesianAgent, self).episode_over(idx)

    def get_action(self, idx):
        pass

    def set_state(self, idx, state):
        super(SingleTaskBayesianAgent, self).set_state(idx, state)

    def observe_reward(self, idx, r):
        super(SingleTaskBayesianAgent, self).observe_reward(idx, r)
        self.model.add_reward_observation(Observation(idx, ))

    def sample_mdp(self):
        # TODO: sample a MAP MDP
        return np.zeros(self.num_colors * NUM_RELATIVE_CELLS)
