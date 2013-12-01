import numpy as np
import math
import random
from scipy.stats import chi2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class NormalInverseWishartDistribution(object):
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)
        self.cholesky = np.linalg.cholesky(self.inv_psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand())
        return (np.random.multivariate_normal(self.mu, sigma / float(self.lmbda)), sigma)

    def wishartrand(self):
        dim = self.inv_psi.shape[0]
        foo = np.zeros((dim,dim))

        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i,j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
                else:
                    foo[i,j]  = np.random.normal(0,1)
        return np.dot(self.cholesky, np.dot(foo, np.dot(foo.T, self.cholesky.T)))

    def posterior(self, data):
        n = len(data)
        mean_data = np.mean(data, axis=0)
        sum_squares = np.sum([np.array(np.matrix(x - mean_data).T * np.matrix(x - mean_data)) for x in data], axis=0)
        mu_n = (self.lmbda * self.mu + n * mean_data) / (self.lmbda + n)
        lmbda_n = self.lmbda + n
        nu_n = self.nu + n
        print 'nu_n: {0}'.format(nu_n)
        print 'Sum of squares: {0}'.format(sum_squares)
        print 'Multiplier: {0}'.format(self.lmbda * n / float(self.lmbda + n))
        print 'Deviation square: {0}'.format(np.array(np.matrix(mean_data - self.mu).T * np.matrix(mean_data - self.mu)))
        psi_n = self.psi + sum_squares + self.lmbda * n / float(self.lmbda + n) * np.array(np.matrix(mean_data - self.mu).T * np.matrix(mean_data - self.mu))
        return NormalInverseWishartDistribution(mu_n, lmbda_n, nu_n, psi_n)

x = NormalInverseWishartDistribution(np.array([0,0])-3,1.,4.,np.eye(2))
samples = [x.sample() for _ in range(10000)]
data = [np.random.multivariate_normal(mu,cov) for mu,cov in samples]
y = NormalInverseWishartDistribution(np.array([0,0]),1.,4.,np.eye(2))
z = y.posterior(data)

print 'mu_n: {0}'.format(z.mu)

print 'psi_n: {0}'.format(z.psi)
print ''
print 'Some samples:'
for i in range(5):
    print z.sample()
    print ''

var_samples = np.array([variance for mean,variance in samples])
print 'True: {0}'.format(x.psi)
print 'max: {0}'.format(max(var_samples[0,0]))
print 'mean: {0}'.format(np.mean(var_samples, axis=0))
#n, bins, patches = plt.hist(var_samples, 50, facecolor='green', alpha=0.5)
#plt.plot(bins)
#plt.show()