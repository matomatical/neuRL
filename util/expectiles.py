import numpy as np


def expectile(sample, tau):
    """
    Compute the tau-th expectile from a sample.

    Parameters:
        sample (array-like): sample to compute expectile of. Should have
            multiple (>1) distinct points.
        tau (float or array-like of floats): asymmetric ratio (or ratios)
            of expectiles to compute, each between 0 and 1 inclusive.

    Returns:
        expectiles (np.ndarray): vector of computed expectiles.
    """
    if np.isscalar(tau): tau = [tau]

    sorted_sample = np.sort(sample)
    n = sorted_sample.size
    # sample cumulative density function
    F = np.arange(1, n+1)/n
    # sample partial moment function
    M = np.cumsum(sorted_sample)/n
    # mean
    m = M[-1]
    # 'candidate' expectiles (points where M[i], F[i] change)
    e = sorted_sample
    
    # for each tau t:
    # find the e where this equation is satisfied:
    # 0 = (1-t)M(e) + t(m-M(e)) - e((1-t)F(e) + t(1-F(e)))
    # TODO: vectorise
    expectiles = np.ndarray(len(tau))
    for j, t in enumerate(tau):
        if t == 1:
            expectiles[j] = e[-1]
            continue
        if t == 0:
            expectiles[j] = e[0]
            continue
        # find point where (neg) gradient changes from positive to negative
        # (i = index of final positive imbalance)
        imbalance = (1-t)*M + t*(m-M) - e*((1-t)*F + t*(1-F))
        i = (imbalance > 0).nonzero()[0][-1]
        if imbalance[i] == 0:
            e_star = e[i]
        elif imbalance[i+1] == 0:
            e_star = e[i+1]
        else:
            e_star = ((1-t)*M[i] + t*(m-M[i])) / ((1-t)*F[i] + t*(1-F[i]))
        expectiles[j] = e_star
    
    return expectiles


def tauspace(k):
    """
    Return k evenly spaced floats between 0 and 1 (not inclusive),
    with the (k//2)th equal to 0.5. k must be odd.
    """
    if not k % 2: raise ValueError("k must be odd.")
    e = 1/(2*k)
    return np.linspace(e, 1-e, k)