import numpy as np
from scipy.stats import chi2

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

x = NormalInverseWishartDistribution(np.array([0,0])-3,1.,3.,np.eye(2))
samples = [x.sample() for _ in range(100000)]
var_samples = np.array([variance for mean,variance in samples])
print 'mean: {0}'.format(np.mean(var_samples, axis=0))