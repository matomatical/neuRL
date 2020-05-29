import numpy as np

def expectile(sample, tau, interpolation=True):
    """
    Compute the tau-th expectile from a sample.

    Parameters:
        sample (array-like): sample to compute expectiles of
        tau (float or array-like of floats): asymmetric ratio or ratios
            of expectiles to compute, each between 0 and 1 inclusive
        interpolation (bool): True for computing exact expectile between
            two sample points, False to return previous sample point
            (default True).

    Returns:
        expectiles (np.ndarray): vector of computed expectiles
    """
    sorted_sample = np.sort(sample)
    n = len(sample)
    M = np.cumsum(sorted_sample/n)
    F = np.cumsum(np.ones_like(sorted_sample)/n)
    m = M[-1] # mean
    # candidate expectiles (where M[i], F[i] change):
    e = sorted_sample
    # for each t:
    # find the e where this equation is satisfied:
    # 0 = (1-t)M(e) + t(m-M(e)) - e((1-t)F(e) + t(1-F(e)))
    if np.isscalar(tau): tau = [tau]
    expectiles = np.ndarray(len(tau))
    for j, t in enumerate(tau):
        imbalance = (1-t)*M + t*(m-M) - e*((1-t)*F + t*(1-F))
        i = np.argmin(np.abs(imbalance))
        if t == 1:
            e_star = e[-1]
        elif not interpolation or imbalance[i] == 0:
            e_star = e[i]
        # optionally, refine answer to *exact* expectile in case it's
        # between two sample points:
        elif imbalance[i] > 0:
            # go a little further!
            e_star = ((1-t)*M[i] + t*(m-M[i])) / ((1-t)*F[i] + t*(1-F[i]))
        else:
            # go back a little bit!
            e_star = ((1-t)*M[i-1]+t*(m-M[i-1])) / ((1-t)*F[i-1]+t*(1-F[i-1]))
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