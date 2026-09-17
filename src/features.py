from __future__ import annotations

import numpy as np
import pandas as pd

from .data import download_vix


FEATURE_COLUMNS = [
    "rv_5",
    "rv_20",
    "rv_60",
    "rv_ratio",
    "momentum_5",
    "momentum_20",
    "ma_distance_20",
    "drawdown_252",
    "volume_z",
    "vix_level",
    "vix_change_5",
]


def _zscore(x, window=60):
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / std.replace(0, np.nan)


def build_features(price: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    out = price.copy()
    out["return"] = np.log(out["close"] / out["close"].shift(1))

    out["rv_5"] = out["return"].rolling(5).std() * np.sqrt(252)
    out["rv_20"] = out["return"].rolling(20).std() * np.sqrt(252)
    out["rv_60"] = out["return"].rolling(60).std() * np.sqrt(252)
    out["rv_ratio"] = out["rv_5"] / out["rv_60"]

    out["momentum_5"] = out["close"].pct_change(5)
    out["momentum_20"] = out["close"].pct_change(20)

    ma = out["close"].rolling(20).mean()
    out["ma_distance_20"] = out["close"] / ma - 1

    peak = out["close"].rolling(252, min_periods=1).max()
    out["drawdown_252"] = out["close"] / peak - 1

    out["volume_z"] = _zscore(np.log1p(out["volume"]))

    vx = vix[["close"]].rename(columns={"close": "vix_level"})
    vx["vix_change_5"] = vx["vix_level"].pct_change(5)

    out = out.join(vx, how="left").ffill()
    return out


def add_forward_regime(data: pd.DataFrame, horizon=5) -> pd.DataFrame:
    out = data.copy()

    future = out["return"].shift(-1)
    forward_vol = future.rolling(horizon).std().shift(-(horizon - 1)) * np.sqrt(252)
    out["forward_vol"] = forward_vol

    # Expanding thresholds use only information available before each date.
    history = out["forward_vol"].shift(1)
    low = history.expanding(min_periods=252).quantile(1 / 3)
    high = history.expanding(min_periods=252).quantile(2 / 3)

    out["regime"] = np.select(
        [out["forward_vol"] <= low, out["forward_vol"] >= high],
        [0, 2],
        default=1,
    )

    return out.dropna(subset=FEATURE_COLUMNS + ["regime"])
