from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def greeks(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    sqrtT = np.sqrt(T)

    return {
        "delta": norm.cdf(d1),
        "gamma": pdf / (S * sigma * sqrtT),
        "vega": S * pdf * sqrtT,
        "theta": -(S * pdf * sigma) / (2 * sqrtT)
        - r * K * np.exp(-r * T) * norm.cdf(d2),
    }
