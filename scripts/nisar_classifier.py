# NISAR L-band water classifier: per-scene quadratic discriminant (HH, HV dB) trained on reference areas
import numpy as np
from scipy import ndimage


def smooth_db(db, valid, k=5):
    lin = np.where(valid, 10 ** (db / 10), 0).astype(np.float32)
    num = ndimage.uniform_filter(lin, k); den = ndimage.uniform_filter(valid.astype(np.float32), k)
    return np.where(valid, 10 * np.log10(np.maximum(num / np.maximum(den, 1e-6), 1e-7)), np.nan).astype(np.float32)


def fit_predict(hh, hv, valid, stable, n=200000, seed=0):
    """stable: 1 = stable land, 2 = stable water. Returns the water probability (0..1) and class parameters."""
    rng = np.random.default_rng(seed)
    X = np.stack([hh, hv], -1)
    params = {}
    for c in (1, 2):
        idx = np.flatnonzero((stable.ravel() == c) & valid.ravel())
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        S = X.reshape(-1, 2)[idx]
        mu = S.mean(0); cov = np.cov(S.T) + np.eye(2) * 1e-3
        params[c] = (mu, cov)
    def loglik(c):
        mu, cov = params[c]; inv = np.linalg.inv(cov); d = X - mu
        return -0.5 * np.einsum("...i,ij,...j->...", d, inv, d) - 0.5 * np.log(np.linalg.det(cov))
    l1, l2 = loglik(1), loglik(2)
    p = 1 / (1 + np.exp(np.clip(l1 - l2, -50, 50)))  # equal priors
    p = np.where(valid, p, np.nan).astype(np.float32)
    return p, {("land" if c == 1 else "water"): dict(mean=[round(float(v), 2) for v in params[c][0]]) for c in params}
