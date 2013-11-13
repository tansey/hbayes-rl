import numpy as np
from gridworld import *
from mdp_solver import *
import random
from scipy.stats import chi2

class MdpClass(object):
    def __init__(self, class_id, weights_mean, weights_cov):
        self.class_id = class_id
        self.weights_mean = weights_mean
        self.weights_cov = weights_cov
        self.inv_weights_cov = np.linalg.inv(weights_cov)

    def likelihood(self, weights):
        multiplier = 1./math.sqrt((2.*math.pi)**self.weights_cov.shape[0]*np.linalg.det(self.weights_cov))
        exponent = -0.5 * np.dot(np.dot(np.transpose(weights - self.weights_mean), self.inv_weights_cov), weights - self.weights_mean)
        return multiplier * math.exp(exponent)

    def sample(self):
        return np.random.multivariate_normal(self.weights_mean, self.weights_cov)

    def sample_posterior(self, states, rewards):
        """
        We have the product of two Gaussians, so we can derive a closed form update for the posterior.
        """
        y = self.inv_weights_cov + np.dot(np.transpose(states), states)
        post_cov = np.linalg.inv(y)
        post_mean = np.dot(np.linalg.inv(y), np.dot(self.inv_weights_cov, self.weights_mean) + np.dot(np.transpose(states), rewards))
        return np.random.multivariate_normal(post_mean, post_cov)

class NormalInverseWishartDistribution(object):
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = lmbda
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        return (np.random.multivariate_normal(self.mu, self.inv_psi / self.lmbda), np.linalg.inv(self.wishartrand()))
 
    def wishartrand(self):
        dim = self.inv_psi.shape[0]
        chol = np.linalg.cholesky(self.inv_psi)
        foo = np.zeros((dim,dim))
        
        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i,j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
                else:
                    foo[i,j]  = np.random.normal(0,1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

class LinearGaussianRewardModel(object):
    """
    A model of the rewards for experiment 1 in the Wilson et al. paper. See section 4.4 for implementation details.
    """
    def __init__(self, num_colors, reward_stdev, classes, assignments, auxillary_distribution, alpha=1., m=2, burn_in=100, mcmc_samples=500, thin=10):
        self.weights_size = num_colors * NUM_RELATIVE_CELLS
        self.reward_stdev = reward_stdev
        self.classes = classes
        self.assignments = assignments
        self.total_mpds = sum(assignments) + 1
        self.auxillary_distribution = auxillary_distribution
        self.alpha = alpha
        self.m = m
        self.burn_in = burn_in
        self.mcmc_samples = mcmc_samples
        self.thin = thin
        assert(len(classes) == len(assignments))
        self.map_class = random.choice(classes) # TODO: Weight by prior likelihood?
        self.weights = self.map_class.sample()
        self.states = []
        self.rewards = []

    def add_observation(self, state, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.update_beliefs()

    def update_beliefs(self):
        states = np.array(self.states)
        rewards = np.array(self.rewards)
        samples = np.zeros(len(self.classes)+1)
        c = random.randrange(len(self.classes))
        mdp_class = self.classes[c]
        w = mdp_class.sample_posterior(states, rewards)
        for i in range(self.mcmc_samples):
            mdp_class = self.sample_assignment(w)
            w = mdp_class.sample_posterior(states, rewards)
            if i >= self.burn_in and i % self.thin == 0:
                # TODO: Do we really just want the chain of assignments?
                # Wouldn't the Bayes estimate here be to average w?
                samples[mdp_class.class_id] += 1
                assignment_probs = [self.classes[i].likelihood(w) for i in range(len(self.classes))]
        print 'Assignment Distribution: {0}'.format(samples)
        # MAP calculations
        map_c = np.argmax(samples)
        if map_c >= len(self.classes):
            self.map_class = self.sample_auxillary()
        else:
            self.map_class = self.classes[map_c]
        self.weights = self.sample_weights(states, rewards)

    def sample_assignment(self, weights):
        """
        Implements Algorithm 3 from the Wilson et al. paper.
        """
        classes = [c for c in self.classes] # duplicate classes
        # Calculate likelihood of assigning to a known class
        assignment_probs = [self.assignments[i] / (self.total_mpds - 1. + self.alpha) * self.classes[i].likelihood(weights) for i in range(len(self.classes))]
        # Calculate likelihood of assigning to a new, unknown class with the default prior
        auxillaries = []
        for i in range(self.m):
            aux = self.sample_auxillary(len(self.classes) + i)
            assignment_probs.append(self.alpha / float(self.m) / (self.total_mpds - 1. + self.alpha) * aux.likelihood(weights))
            classes.append(aux) # add auxillary classes to the list of options

        # Sample an assignment proportional to the likelihoods
        partition = sum(assignment_probs)
        assignment_probs = [x / partition for x in assignment_probs]
        u = random.random()
        cur = 0.
        for i,prob in enumerate(assignment_probs):
            cur += prob
            if u <= cur:
                return classes[i] 

    def sample_auxillary(self, class_id):
        (mean, cov) = self.auxillary_distribution.sample()
        return MdpClass(class_id, mean, cov)

    def sample_weights(self, states, rewards):
        self.weights = self.map_class.sample_posterior(states, rewards)

class MultiTaskBayesianAgent(Agent):
    """
    A Bayesian RL agent that infers a hierarchy of MDP distributions, with a top-level
    class distribution which parameterizes each bottom-level MDP distribution.
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

if __name__ == "__main__":
    TRUE_CLASS = 0
    SAMPLE_SIZE = 100
    COLORS = 2
    RSTDEV = 0.3
    SIZE = COLORS * NUM_RELATIVE_CELLS
    NUM_DISTRIBUTIONS = 4

    niw_true = NormalInverseWishartDistribution(np.zeros(SIZE) - 3., 1., SIZE+1, np.identity(SIZE))
    true_params = [niw_true.sample() for _ in range(NUM_DISTRIBUTIONS)]
    classes = [MdpClass(i, mean, cov) for i,(mean,cov) in enumerate(true_params)]
    assignments = [1. for _ in classes]
    #default_class = MdpClass(-1, np.zeros(COLORS * NUM_RELATIVE_CELLS), np.identity(COLORS * NUM_RELATIVE_CELLS))
    auxillary = NormalInverseWishartDistribution(np.zeros(SIZE), 1., SIZE+1, np.identity(SIZE)*5)

    model = LinearGaussianRewardModel(COLORS, RSTDEV, classes, assignments, auxillary)

    weights = classes[TRUE_CLASS].sample()

    print 'True class: {0}'.format(TRUE_CLASS)

    for s in range(SAMPLE_SIZE):
        q_sample = np.zeros((COLORS * NUM_RELATIVE_CELLS))
        for row in range(NUM_RELATIVE_CELLS):
            q_sample[row * COLORS + random.randrange(COLORS)] = 1
        r_sample = np.random.normal(loc=np.dot(weights, q_sample), scale=RSTDEV)
        model.add_observation(q_sample, r_sample)
        print 'Samples: {0} Class belief: {1}'.format(s+1, model.map_class.class_id)