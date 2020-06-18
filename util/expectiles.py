"""
Function to compute expectiles of a sample.

Matthew Farrugia-Roberts, 2020
"""

import numpy as np


def expectile(sample, tau, sorted_sample=False):
    """
    Compute the tau-th expectile from a sample.

    Parameters:
        sample (array-like): sample to compute expectile of. Should have
            multiple (>1) distinct points. If all the points are the same,
            all of the expectiles are just equal to the points.
        tau (float or array-like of floats): asymmetric ratio (or ratios)
            of expectiles to compute, each between 0 and 1 inclusive.
        sorted_sample (bool): Mark sample as already sorted. Default: False,
            assumes sample is not sorted, sorts sample into a new array.

    Returns:
        expectiles (float or np.ndarray of floats): vector of computed
            expectiles (or scalar if tau was scalar).
    """
    # preprocess input
    scalar = np.isscalar(tau)
    if scalar:
        tau = np.array([tau])
    else:
        tau = np.asarray(tau)
    if np.any(tau < 0) or np.any(tau > 1):
        raise ValueError("All tau values must be between 0 and 1.")

    # sort sample if necessary
    if sorted_sample:
        x = np.asarray(sample)
    else:
        x = np.sort(sample)
    
    # precompute F and M at key points
    # (for stability, omit division by N)
    N = x.size
    # N * sample cumulative density function
    F = np.arange(N) + 1
    # N * sample partial moment function
    M = np.cumsum(x)
    # N * sample mean
    MN = M[-1]
    # N * 1
    FN = N # = F[-1]

    # prepare output array
    eps = np.ndarray(len(tau))

    # solve all edge cases first
    eps[tau == 0] = x[0]
    eps[tau == 1] = x[-1]

    # then solve remaining cases
    j = np.where((tau != 0) & (tau != 1))
    t = tau[j][:, np.newaxis]
    A = -((1-2*t)*F + t*FN)
    B =  ((1-2*t)*M + t*MN)
    G = A*x + B
    # find the segment with G_i >= 0 and G_i+1 < 0:
    # (pad True should catch case when all are same)
    i = np.where((G >= 0)
        & np.pad(G[:, 1:] < 0, [(0, 0), (0, 1)], constant_values=True))
    # interpolate to get the root (unless it's an exact sample point)
    eps[j] = np.where(G[i] == 0, x[i[1]], -B[i]/A[i])

    # postprocess and return output
    if scalar:
        return eps[0]
    else:
        return eps


def tauspace(k, edges=False):
    """
    Construct an array of `k` evenly spaced floats between 0 and 1
    (not inclusive, unless `edges` is set to True).
    `k` must be odd, so the `k//2`th element of the array is 0.5.
    
    Parameters:
        k (int): Number of floats to include. Must be odd.
        edges (bool): Whether to include edges. If so, the returned
            array starts at 0 and ends at 1. If not, the returned
            array starts and ends half an interval away from 0 and 1,
            compared to the difference between successive elements.
            Default: False.
    
    Returns:
        taus (np.ndarray): array of evenly spaced floats. Shape: (k,).
    """
    if not k % 2: raise ValueError("k must be odd.")
    e = 0 if edges else 1/(2*k)
    return np.linspace(e, 1-e, k)
