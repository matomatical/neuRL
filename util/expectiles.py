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
    eps = np.empty_like(tau)

    # solve all edge cases first
    eps[tau == 0] = x[0]
    eps[tau == 1] = x[-1]

    # then solve remaining cases
    k = np.where((tau != 0) & (tau != 1))
    t = tau[k][:, np.newaxis]
    # compute line segments
    A = -((1-2*t)*F + t*FN) # slopes
    B =  ((1-2*t)*M + t*MN) # offsets
    G = A*x + B
    # for each i find the segment with G_i,j >= 0 and G_i,j+1 < 0:
    i, j = np.where((G >= 0)[:, :-1] & (G < 0)[:, 1:])
    # due to numerical issues there may not be eactly one j for each
    # possible i. if no j for some i, use N-1. if many j, use last one.
    I = np.arange(t.size)
    J = np.full_like(I, N-1)
    J[i] = j # overwrite all but the last j or the default for each i
    
    # interpolate to get the root (unless it's an exact sample point,
    # in which case we do better by just taking the sample point itself.)
    eps[k] = np.where(G[I, J] == 0, x[J], -B[I, J]/A[I, J])

    # postprocess and return output
    if scalar:
        return eps[0]
    else:
        return eps


def tauspace(k, endpoints=False):
    """
    Construct an array of `k` evenly spaced floats between 0 and 1
    (not inclusive, unless `endpoints` is set to True).
    `k` must be odd, so the `k//2`th element of the array is 0.5.
    
    Parameters:
        k (int): Number of floats to include. Must be odd.
        endpoints (bool): Whether to include edges. If so, the returned
            array starts at 0 and ends at 1. If not, the returned
            array starts and ends half an interval away from 0 and 1,
            compared to the difference between successive elements.
            Default: False.
    
    Returns:
        taus (np.ndarray): array of evenly spaced floats. Shape: (k,).
    """
    if not k % 2: raise ValueError("k must be odd.")
    e = 0 if endpoints else 1/(2*k)
    return np.linspace(e, 1-e, k)
