from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_history(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No historical data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def download_vix(start: str = "2010-01-01") -> pd.DataFrame:
    return download_history("^VIX", start)


def current_option_chain(ticker: str):
    """Return real current option-chain snapshots from Yahoo Finance."""
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        raise ValueError("No option expirations are available.")

    rows = []
    for expiry in expirations[:12]:
        chain = tk.option_chain(expiry)

        for side, frame in [("call", chain.calls), ("put", chain.puts)]:
            if frame is None or frame.empty:
                continue
            x = frame.copy()
            x["option_type"] = side
            x["expiration"] = expiry
            rows.append(x)

    if not rows:
        raise ValueError("No option contracts were returned.")

    return pd.concat(rows, ignore_index=True)
