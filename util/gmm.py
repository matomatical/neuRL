import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class GMM:
    """
    Scalar Gaussian Mixture Model

    Parameters:
        coeffs (array-like): array of mixture coefficients (shape: (k,))
        params (np.ndarray): array of corresponding location (mean), scale
            (standard deviation) parameters (shape: (k, 2)) 
    """
    def __init__(self, coeffs, params):
        self.coeffs_params = np.column_stack([coeffs, params])
        self.coeffs = self.coeffs_params[:, 0]
        self.params = self.coeffs_params[:, 1:]
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
            which_samples = np.where(which_gaussian==i)
            count_samples = len(which_samples[0])
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
        return self.coeffs @ [norm.pdf(x,μ,σ) for μ, σ in self.params]
    
    def mean(self):
        """
        Compute model's mean.
        
        Returns:
            μ (float): mean of the mixture model.
        """
        return self.coeffs @ self.params[:, 0]
        
    
    def __str__(self):
        return "+".join(["{:.3f}*N(μ={:.1f},σ={:.1f})".format(*d) for d in self.coeffs_params])
    def __repr__(self):
        return "GMM(coeffs={}, params={})".format(self.coeffs, self.params)


def EXAMPLE(n=1000, coeffs=[0.1, 0.4, 0.5], params=[[7, 1], [-1, 2], [-10, 5]], seed=32,
            plot=True, figsize=(13, 7)):
    # create a mixture of gaussians
    gmm = GMM(coeffs=coeffs, params=params)

    # draw and a sample of 1000 points for us to play with
    np.random.seed(seed=seed)
    sample = gmm.rvs(n)

    if plot:
        # plot the sample
        plt.figure(figsize=figsize)
        plt.hist(sample, density=1, bins=100, alpha=0.5, label="sample histogram")
        plt.scatter(sample, np.zeros_like(sample), marker="|", s=100, label="sample points")
        # plot pdf
        x = np.linspace(sample.min(), sample.max(), 300)
        y = gmm.pdf(x)
        plt.plot(x, y, color="black", label="pdf")

        plt.xlabel("x")
        plt.ylabel("density")
        plt.legend()
        plt.show()
        
        return sample, x, y

    else:
        return sample