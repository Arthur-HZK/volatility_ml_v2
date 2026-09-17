from __future__ import annotations

import numpy as np
import pandas as pd


def run_regime_backtest(
    price_data: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float = 0.55,
):
    """Simple defensive regime signal.

    Signal = 1 when predicted probability of high volatility is below
    the threshold; otherwise 0. Returns are shifted so today's signal
    affects the following trading day.
    """
    df = price_data[["close"]].join(
        predictions[["prob_High"]],
        how="inner",
    )

    df["return"] = df["close"].pct_change()

    df["position"] = (df["prob_High"] < threshold).astype(float)
    df["strategy_return"] = df["position"].shift(1) * df["return"]

    df["buy_hold_curve"] = (1 + df["return"].fillna(0)).cumprod()
    df["strategy_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df


def performance_stats(df):
    r = df["strategy_return"].dropna()
    if len(r) == 0:
        return {}

    ann_return = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    curve = (1 + r).cumprod()
    drawdown = curve / curve.cummax() - 1

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "observations": len(r),
    }
