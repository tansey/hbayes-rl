import numpy as np
from numpy.random import standard_normal, chisquare, multivariate_normal, dirichlet, multinomial
from numpy.linalg import cholesky, inv
from math import sqrt
import math

class GibbsSampler(object):
    """Gibbs sampler for finite Gaussian mixture model

    Given a set of hyperparameters and observations, run Gibbs sampler to estimate the parameters of the model
    """
    def __init__(self, hyp_pi, mu0, kappa0, T0, nu0, y, prior_z):

        """Initialize the Gibbs sampler

        @para hyp_pi: hyperparameter of pi
        @para mu0, kappa0: parameter of Normal-Wishart distribution
        @para T0, nu0: parameter of Normal-Wishart distribution
        @para y: samples draw from Normal distributions
        """
        self.hyp_pi = hyp_pi
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.T0 = T0
        self.nu0 = nu0
        self.y = y
        self.prior_z = prior_z

    def _clusters(self):
        return len(self.hyp_pi)

    def _dim(self):
        """
        Dimension of the data
        """
        return len(mu0)

    def _numbers(self):
        return len(self.y)

    def estimate_clusters(self, iterations, burn_in, lag):
        """
        """
        estimated_clusters = np.zeros(self._numbers(), float)
        for iteration, z, pi, mu, sigma in self.run(iterations, burn_in, lag):
            #  print "Precision = %f" % self._estimate_precision(z)
            count = [z.count(k) for k in range(self._clusters())]
            print count
            if iteration % 10 == 0:
                print "iteration = %d" %iteration

    def _estimate_precision(self, z):
        numbers = self._numbers()
        count = 0
        for i in range(numbers):
            if self.prior_z[i] == z[i]:
                count += 1
        return float(count)/float(numbers)

    def run(self, iterations, burn_in, lag):
        """
        Run the Gibbs sampler
        """
        self._initialize_gibbs_sampler()
        lag_counter = lag
        iteration = 1
        while iteration <= iterations:
            self._iterate_gibbs_sampler()
            if burn_in > 0:
                burn_in -= 1
            else:
                if lag_counter > 0:
                    lag_counter -= 1
                else:
                    lag_counter = lag
                    yield iteration, self.z, self.pi, self.mu, self.sigma
                    iteration += 1

    def _initialize_gibbs_sampler(self):
        """
        This sets the initial values of the parameters.
        """
        clusters = self._clusters()
        numbers = self._numbers()
        self.mu = np.array([self._sampling_Normal_Wishart()[0] for _ in range(clusters)])
        self.pi = dirichlet(hyp_pi, 1)[0]
        self.sigma = np.array([self._sampling_Normal_Wishart()[1] for _ in range(clusters)])
        self.z = np.array([self._multinomial_samples(pi) for _ in range(numbers)])

    def _sampling_Normal_Wishart(self):
        """
        Sampling mu and sigma from Normal-Wishart distribution.

        """
        # Create the matrix A of the Bartlett decomposition from a p-variate Wishart distribution
        d = self._dim()
        chol = np.linalg.cholesky(self.T0)

        if (self.nu0 <= 81+d) and (self.nu0 == round(self.nu0)):
            X = np.dot(chol, np.random.normal(size = (d, self.nu0)))
        else:
            A = np.diag(np.sqrt(np.random.chisquare(self.nu0 - np.arange(0, d), size = d)))
            A[np.tri(d, k=-1, dtype = bool)] = np.random.normal(size = (d*(d-1)/2.))
            X = np.dot(chol, A)
        inv_sigma = np.dot(X, X.T)
        mu = np.random.multivariate_normal(self.mu0, np.linalg.inv(self.kappa0*inv_sigma))

        return mu, np.linalg.inv(inv_sigma)

    def _norm_pdf_multivariate(self, index, cluster):
        """
        Calculate the probability density of multivariable normal distribution
        """
        d = self._dim()
        m = self.y[index] - self.mu[cluster]
        part1 = np.dot(m, np.linalg.inv(self.sigma[cluster]))
        part = np.dot(part1, m.T)
        value = 1.0 / (math.pow(2.0*math.pi, d*0.5) * math.sqrt(np.linalg.det \
            (self.sigma[cluster]))) * math.exp(-(0.5)*part)
        return value

    def _iterate_gibbs_sampler(self):
        """
        Updates the values of the z, pi, mu, sigma.
        """
        clusters = self._clusters()
        # sampling the indicator variables
        pos_z = []
        for i in range(len(self.y)):
            f_xi = np.array([self._norm_pdf_multivariate(i, k) for k in range(clusters)])
            prob_zi = (self.pi * f_xi) / np.dot(self.pi, f_xi)
            pos_zi = self._multinomial_samples(prob_zi)
            pos_z.append(pos_zi)

        # sampling new mixture weights
        count_k = np.array([pos_z.count(k) for k in range(clusters)])
        pos_pi = np.random.dirichlet(count_k + self.pi, 1)[0]

        # sampling parameters for each cluster
        pos_x = []
        for k in range(clusters):
            pos_xk = np.array([self.y[i] for i in range(len(pos_z)) if pos_z[i] == k ])
            pos_x.append(pos_xk)
        # calculate the posterior of multi-normal distribution
        pos_mu = []
        pos_sigma = []
        for k in range(clusters):
            if len(pos_x[k]) == 0: # No observations, no update.
                pos_T0 = self.T0
                pos_mu0 = self.mu0
                pos_kappa0 = self.kappa0
                pos_nu0 = self.nu0
            else:
                # Update the parameters of Normal-Wishart distribution.
                # mean_k is the sample mean in k-th cluster.
                # C is the sample covariance matrix.
                # D is the true covariance matrix.
                mean_k = np.mean(pos_x[k], axis=0)
                C = np.zeros((len(mean_k), len(mean_k)))
                for x_i in pos_x[k]:
                    C += (x_i - mean_k).reshape(len(mean_k), 1) * (x_i - mean_k)
                pos_nu0 = self.nu0 + len(pos_x[k])
                pos_kappa0 = self.kappa0 + len(pos_x[k])
                D = float(self.kappa0  * len(pos_x[k])) / (self.kappa0 + len(pos_x[k])) * \
                    (mean_k - self.mu0).reshape(len(mean_k), 1) * (mean_k - self.mu0)
                pos_T0 = np.linalg.inv(np.linalg.inv(self.T0) + C + D)
                pos_mu0 = (self.kappa0*self.mu0 + len(pos_x[k])*mean_k) / (self.kappa0 + len(pos_x[k]))
                # Update posterior parameters of Normal-Wishart distribution.
                # Then draw the new parameters pos_mu and pos_sigma for each cluster.
            self.mu0 = pos_mu0
            self.kappa0 = pos_kappa0
            self.T0 = pos_T0
            self.nu0 = pos_nu0
            pos_mu_k, pos_sigma_k = self._sampling_Normal_Wishart()
            print pos_mu_k
            pos_mu.append(pos_mu_k)
            pos_sigma.append(pos_sigma_k)

        # After all parameters updated, pass them to the initial values.
        self.z = pos_z
        self.pi = pos_pi
        self.mu = pos_mu
        self.sigma = pos_sigma

    def _multinomial_samples(self, distributions):
        return np.nonzero(multinomial(1, distributions))[0][0]

def multinomial_sample(distributions):
    return np.nonzero(multinomial(1, distributions))[0][0]

def generate_observations(clusters, numbers, hyp_pi = None):
    if hyp_pi == None:
        hyp_pi = [1]*clusters
    pi = dirichlet(hyp_pi, 1)[0]
    mu = []
    sigma = []
    observations = []
    prior_z = []
    for i in range(clusters):
        m, s = sampling_Normal_Wishart(mu0, kappa0, T0, nu0)
        mu.append(m)
        sigma.append(s)
    for i in range(clusters):
        cluster = multinomial_sample(pi)
        obs = multivariate_normal(mu[cluster], sigma[cluster], k_num)
        observations.extend(list(obs))
        prior_z.extend([cluster]*k_num)
    return observations, prior_z

def sampling_Normal_Wishart(mu0, kappa0, T0, nu0):
    """
    Sampling cluster parameters from normal inverse Wishart distribution.
    """
    d = len(mu0)
    chol = np.linalg.cholesky(T0)

    if (nu0 <= 81+d) and (nu0 == round(nu0)):
        X = np.dot(chol, np.random.normal(size = (d, nu0)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(nu0 - np.arange(0, d), size = d)))
        A[np.tri(d, k=-1, dtype = bool)] = np.random.normal(size = (d*(d-1)/2.))
        X = np.dot(chol, A)
    inv_sigma = np.dot(X, X.T)
    mu = np.random.multivariate_normal(mu0, np.linalg.inv(kappa0*inv_sigma))
    return mu, np.linalg.inv(inv_sigma)

if __name__ == "__main__":
    # Generate the data set.
    # Initialize the parameters for the model.
    # d: dimension of the data.
    # mu0, kappa0, T0, nu0 are the parameters of the Normal-Wishart distribution.
    kappa0 = 4.0
    d = 2
    T0 = np.diag(np.ones(d))
    mu0 = np.zeros(d)
    nu0 = 14.0
    clusters = 6
    k_num = 50
    hyp_pi = [1]*clusters
    pi = dirichlet(hyp_pi, 1)[0]
    y, prior_z = generate_observations(clusters, k_num, hyp_pi = None)
    prior_count = [prior_z.count(k) for k in range(clusters)]
    sampler = GibbsSampler(hyp_pi, mu0, kappa0, T0, nu0, y, prior_z)
    sampler.estimate_clusters(200, 3, 0)