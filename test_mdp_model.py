import numpy as np
import random

class MdpClass(object):
    def __init__(self, class_id, weights_mean, weights_cov):
        self.class_id = class_id
        self.weights_mean = weights_mean
        self.weights_cov = weights_cov
        self.inv_weights_cov = np.linalg.inv(weights_cov)

    def likelihood(self, weights):
        # Note: we ignore the 1./math.sqrt((2.*math.pi)**self.weights_cov.shape[0]) constant
        multiplier = math.pow(np.linalg.det(self.weights_cov), -0.5)
        exponent = -0.5 * np.dot(np.dot(np.transpose(weights - self.weights_mean), self.inv_weights_cov), weights - self.weights_mean)
        if multiplier < 0 or math.exp(exponent) < 0:
            print 'Mean: {0} Cov: {1}'.format(self.weights_mean, self.weights_cov)
            print 'Weights: {0}'.format(weights)
            print 'Multiplier: {0} Exponent: {1}'.format(multiplier, exponent)
            raise Exception('Multiplier or Exponent < 0 in multivariate normal likelihood function.')
        return multiplier * math.exp(exponent)

    def sample(self):
        return np.random.multivariate_normal(self.weights_mean, self.weights_cov)

    def posterior(self, states, rewards):
        """
        We have the product of two Gaussians, so we can derive a closed form update for the posterior.
        """
        y = self.inv_weights_cov + np.dot(np.transpose(states), states)
        post_cov = np.linalg.inv(y)
        post_mean = np.dot(np.linalg.inv(y), np.dot(self.inv_weights_cov, self.weights_mean) + np.dot(np.transpose(states), rewards))
        return MdpClass(self.class_id, post_mean, post_cov)

    def sample_posterior(self, states, rewards):
        return self.posterior(states, rewards).sample()

RELATIVE_CELLS = 5
COLORS = 8
ndims = RELATIVE_CELLS * COLORS
SAMPLE_SIZE = 100

weights = np.ones(ndims) * -10.
reward_sigma = 0.1

states = []
rewards = []
for s in range(SAMPLE_SIZE):
    state = np.zeros((ndims))
    for row in range(RELATIVE_CELLS):
        state[row * COLORS + random.randrange(COLORS)] = 1
    states.append(state)
    rewards.append(random.normalvariate(np.dot(weights, state), reward_sigma))
states = np.array(states)
rewards = np.array(rewards)

mdp = MdpClass(0, np.zeros(ndims), np.eye(ndims))
print 'Prior: {0}'.format(mdp.weights_mean[0:3])
post_mdp = mdp.posterior(states,rewards)
print 'Posterior: {0}'.format(post_mdp.weights_mean[0:3])
print 'Samples:'
for i in range(5):
    print post_mdp.sample()[0:3]