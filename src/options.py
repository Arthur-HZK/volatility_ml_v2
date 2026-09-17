from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .black_scholes import greeks


def clean_option_chain(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    out = chain.copy()

    numeric = [
        "strike", "lastPrice", "bid", "ask", "volume",
        "openInterest", "impliedVolatility"
    ]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["mid"] = (out["bid"] + out["ask"]) / 2

    # Keep contracts with usable observed IV and reasonable liquidity.
    out = out[
        out["strike"].between(0.5 * spot, 1.5 * spot)
        & out["impliedVolatility"].gt(0)
        & out["impliedVolatility"].lt(5)
    ].copy()

    out["moneyness"] = out["strike"] / spot
    out["days_to_expiry"] = (
        pd.to_datetime(out["expiration"]) - pd.Timestamp.utcnow().tz_localize(None)
    ).dt.days.clip(lower=1)

    return out


def option_surface(chain: pd.DataFrame) -> pd.DataFrame:
    cols = ["strike", "days_to_expiry", "impliedVolatility", "option_type", "volume", "openInterest"]
    return chain[[c for c in cols if c in chain]].dropna()


def enrich_selected_contract(
    row: pd.Series,
    spot: float,
    risk_free_rate: float = 0.04,
) -> dict:
    T = max(float(row["days_to_expiry"]) / 365, 1 / 365)
    K = float(row["strike"])
    sigma = float(row["impliedVolatility"])

    g = greeks(spot, K, T, risk_free_rate, sigma)

    result = {
        "strike": K,
        "days_to_expiry": int(round(T * 365)),
        "implied_volatility": sigma,
        "observed_mid": float(row["mid"]) if pd.notna(row["mid"]) else np.nan,
    }
    result.update(g)
    return result
