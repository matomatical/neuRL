#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.special as sc
import scipy.optimize as opt

from util.expectiles import expectile, tauspace


# ## Helper functions

def exp_smooth(x, y, newx=None, ω=1):
    """
    Given x, y(x), compute y_s(newx), smoothed version, with bandwidth ω
    """
    if newx is None: newx=x
    W = np.exp(-np.square(newx[:, np.newaxis] - x)/ω)
    W /= W.sum(axis=-1, keepdims=True)
    return W @ y

def polyfit(x, y, d, **kwargs):
    return np.polynomial.polynomial.Polynomial.fit(x, y, d, **kwargs)

def polyinv(p, y, tol=1e-6, domain=None):
    if np.isscalar(y):
        y = [y]
        n = 1
    else:
        n = len(y)
    x = np.zeros(n)
    for i in range(n):
        xi = (p - y[i]).roots()
        xi = xi[np.abs(xi.imag) < tol].real
        if domain:
            xi = xi[domain[0] < xi]
            xi = xi[xi < domain[1]]
        x[i] = xi[:1]
    return x

def lerp(x0, y0, y):
    """
    Return x in (increasing) x0 corresponding to y in (increasing) y0
    using linear interpolation (or nan for out-of-range values)
    """
    n = len(y0)
    x = np.zeros_like(y, dtype=float)
    i = np.searchsorted(y0, y)
    out = (i == 0) | (i == n)

    ok = ~out
    i = i[ok]
    y_lo = y0[i-1]
    y_hi = y0[i]
    x_lo = x0[i-1]
    x_hi = x0[i]
    t = (y[ok] - y_lo)/(y_hi - y_lo)
    x[out] = np.nan
    x[ok] = (1-t)*x_lo + t*x_hi
    return x


# ## Imputers

class NaiveImputer:
    def fit(self, ε, τ):
        self.ε = ε
        self.τ = τ
        return self
    def sample(self, k):
        return np.random.choice(self.ε, k)
    def __repr__(self):
        return f"NaiveImputer()"


class OptBasedImputer:
    def __init__(self, method='root', start='bestof1000', use_lrm=False, verbose=False):
        self.method = method
        self.start = start
        self.use_lrm = use_lrm
        self.verbose = verbose
    def fit(self, ε, τ, i=None):
        self.τ = τ
        self.ε = ε
        if self.use_lrm:
            self.lrm = np.sqrt(τ / (1-τ))
        else:
            self.lrm = 1
        return self
    def sample(self, k):
        # initialise
        if self.verbose: print("initialising...")
        lo, hi = self.ε.min(), self.ε.max()
#         lo, hi = 1.5*lo - 0.5*hi, 1.5*hi - 0.5*lo # expand initialisation to help optimiser?
        if self.start == 'uniform':
            # initialise as uniform distribution between most extreme expectiles
            z0 = np.linspace(lo, hi, k)
        elif self.start == 'bestof1000':
            # this is implemented for 2020 paper to 'significantly improve optimum found'
            # (note: they also repeat this 10 times)
            zs = np.random.uniform(lo, hi, (1000, k))
            z0 = zs[np.argmin(self._ER_loss(zs))]
        # optimise
        if self.verbose: print("optimising...")
        if self.method=='root':
            if k != len(self.ε):
                raise Exception("Hey! This method requires n=k (number of expectiles)")
            x = opt.root(self._grad_ε_ER_loss, x0=z0).x
        elif self.method=='min':
            # this method is much slower...
            def sum_square_grads(sample):
                return np.sum(self._grad_ε_ER_loss(sample)**2)
            x = opt.minimize(sum_square_grads, x0=z0).x
        return x
    def _ER_loss(self, X):
        n      = X.shape[-1]
        rpe    = X[..., np.newaxis] - self.ε
        scale  = np.abs((rpe < 0) - self.τ)
        losses = np.sum(scale * self.lrm * np.square(rpe) / n, axis=-2)
        return np.sum(losses, axis=-1)
    def _grad_ε_ER_loss(self, X):
        n     = X.size
        rpe   = X[:, np.newaxis] - self.ε  # rpe[i, k] = X[i] - ε[k]
        scale = np.abs((rpe < 0) - self.τ) # scale[i, k] = τ[k] if rpe[i, k] > 0 else 1 - τ[k]
        grads = 2 * np.sum(scale * self.lrm * rpe, axis=0) / n
        return grads
    def __repr__(self):
        return f"OptBasedImputer({self.method=}, {self.start=})"


class DirectImputer:
    def __init__(self, smooth_invert=False, exp_tails=False):
        self.smooth_invert = smooth_invert
        self.exp_tails = exp_tails
    def fit(self, ε_, τ_, i=None):
        g_ = np.gradient(ε_, τ_, edge_order=2)
        if i is None: i = ε_.size//2
        self.μ = μ = ε_[i]
        self.ε = ε = np.delete(ε_, i)
        self.τ = τ = np.delete(τ_, i)
        self.g = g = np.delete(g_, i)
        self.N = N = -(ε - μ + τ * g * (1-2*τ))
        self.D = D = g * (1-2*τ)**2
        self.F = F = N / D
        self.f = np.gradient(F, ε, edge_order=1)
        return self
    def sample(self, k):
        y = np.random.random(k)
        if self.smooth_invert:
            x = exp_smooth(self.F, self.ε, y, ω=1/self.ε.size**2)
        else:
            i = np.searchsorted(self.F, y)
            x = self.ε[np.clip(i, 0, self.ε.size-1)]
        if self.exp_tails:
            lo = np.where(y <  self.F.min())
            x[lo] = self.ε.min() - np.random.exponential(size=len(lo[0]))
            hi = np.where(y >= self.F.max())
            x[hi] = self.ε.max() + np.random.exponential(size=len(hi[0]))
        return x
    def __repr__(self):
        return f"DirectImputer({self.smooth_invert=}, {self.exp_tails=})"


class InterpolatingImputer:
    def fit(self, ε_, τ_, i=None):
        # find CDF for interpolation
        g_ = np.gradient(ε_, τ_, edge_order=2)
        if i is None: i = ε_.size//2
        self.μ = μ = ε_[i]
        self.ε = ε = np.delete(ε_, i)
        self.τ = τ = np.delete(τ_, i)
        self.g = g = np.delete(g_, i)
        self.N = N = -(ε - μ + τ * g * (1-2*τ))
        self.D = D = g * (1-2*τ)**2
        self.F = F = N / D
        self.f = np.gradient(F, ε, edge_order=1)
        # then a linear fit for extrapolation
        self.p = polyfit(ε, sc.logit(F), 1, domain=[-1,1],window=[-1,1])
        self.b, self.a = self.p
        return self
    def F_inv(self, y):
        x = lerp(self.ε, self.F, y)
        out = np.isnan(x)
        x[out] = (sc.logit(y[out])-self.b)/self.a
        return x
    def sample(self, k):
        y = np.random.random(k)
        x = self.F_inv(y)
        return x
    def __repr__(self):
        return f"InterpolatingImputer()"


class PolyLogitInterpImputer:
    def __init__(self, x=np.linspace(-65, 50, 2000), degree=3):
        self.degree = degree
        self.x = x
    def fit(self, ε_, τ_, i=None):
        self.ε_ = ε_
        self.τ_ = τ_
        self.ε = ε = x = self.x
        self.p = p = polyfit(ε_, sc.logit(τ_), self.degree)
        self.μ = μ = polyinv(p, sc.logit(0.5))
        self.τ = τ = sc.expit(p(ε))
        self.g = g = np.gradient(ε, τ, edge_order=2)
        self.N = N = -(ε - μ + τ*g*(1-2*τ))
        self.D = D = g * (1-2*τ)**2
        self.F = F = N / D
        self.f = f = np.gradient(F, ε, edge_order=1)
        return self
    def sample(self, k):
        y = np.random.random(k)
        i = np.searchsorted(self.F, y)
        x = self.ε[np.clip(i, 0, self.ε.size-1)]
        return x
    def __repr__(self):
        return f"PolyLogitInterpImputer({self.degree=})"


class PolyLogitImputer:
    def __init__(self, degree=3, i=None, batch_size=100):
        self.degree = degree
        self.batch_size = batch_size
    def fit(self, ε_, τ_):
        self.ε_  = ε_
        self.τ_  = τ_
        self.p   = polyfit(ε_, sc.logit(τ_), self.degree)
        self.dp  = self.p.deriv()
        self.ddp = self.dp.deriv()
        self.μ   = polyinv(self.p, sc.logit(0.5))
        return self
    def F(self, x):
        μ   = self.μ        # ε(0.5)
        px  = self.p(x)     # p(x)
        dpx = self.dp(x)    # p'(x)
        spx = sc.expit(px)  # σ(p(x))
        tpx = (1-2*spx)     # tanh(p(x)) = 1-2σ(p(x))
        return -spx/tpx*(1 + (x-μ)*(1-spx)*dpx/tpx)
    def f(self, x):
        xμ   = x - self.μ         # x - μ
        px   = self.p(x)          # p(x)
        dpx  = self.dp(x)         # p'(x)
        ddpx = self.ddp(x)        # p''(x)
        sx   = sc.expit(px)       # τ(x) = σ(p(x))
        tx   = (1-2*sx)           # 1-2τ(x) = tanh(p(x))
        dsx  = sx*(1-sx)*dpx      # τ'(x) = σ(p(x))(1-σ(p(x)))p'(x)
        fx = -dsx*(xμ*dpx*tx + xμ*ddpx/dpx + 2 + 4*xμ*dsx/tx)/(tx**2)
        return fx
    def τ(self, x):
        return sc.expit(self.p(x))
    def ε(self, y):
        return polyinv(self.p, sc.logit(y))
    def sample(self, k):
        y = np.random.random(k)
        if self.batch_size is not None:
            bs = [y[i:i+self.batch_size] for i in range(0, k, self.batch_size)]
            xs = [opt.root(lambda x: self.F(x)-b, x0=np.zeros_like(b), jac=lambda x: np.diag(self.f(x))).x for b in bs]
            x = np.concatenate(xs)
        else:
            x = opt.root(lambda x: self.F(x)-y, x0=np.zeros(k), jac=lambda x: np.diag(self.f(x))).x
        return x
    def __repr__(self):
        return f"PolyLogitImputer({self.degree=})"
