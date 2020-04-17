import numpy as np
from scipy.stats import norm

class GMM:
    """
    Scalar Gaussian Mixture Model

    Parameters:
        coeffs (array-like): array of mixture coefficients (shape: (k,))
        params (np.ndarray): array of corresponding location (mean), scale
            (standard deviation) parameters (shape: (k, 2)) 
    """
    def __init__(self, coeffs, params):
        self.coeffs = np.array(coeffs)
        self.params = np.array(params)
        self.K = len(coeffs)
        
    def rvs(self, n):
        """
        Sample n points from the mixture model.

        Parameters:
            n (int): number of points to sample

        Returns:
            samples (np.ndarray): sample points (shape: (n,))
        """
        which_gaussian = np.random.choice(self.K, size=n, p=self.coeffs)
        samples = np.ndarray(n)
        for i, (loc, scale) in enumerate(self.params):
            which_samples = (which_gaussian==i)
            count_samples = np.count_nonzero(which_samples)
            samples[which_samples] = norm.rvs(loc, scale, size=count_samples)
        return samples

    def pdf(self, x):
        """
        Compute mixture model's pdf at point(s) x.

        Parameters:
            x (float or np.ndarray of floats): points to evaluate at

        Returns:
            p (float or np.ndarray of floats): pdf at those points
        """
        mixture = zip(self.coeffs, self.params)
        return sum(c * norm.pdf(x, *ps) for c, ps in mixture)
