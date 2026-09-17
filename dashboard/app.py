from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import performance_stats, run_regime_backtest
from src.data import current_option_chain, download_history, download_vix
from src.features import FEATURE_COLUMNS, add_forward_regime, build_features
from src.model import evaluate, final_model_importance, walk_forward_predict
from src.options import clean_option_chain, option_surface


st.set_page_config(page_title="Quant Volatility Research", layout="wide")
st.title("Quantitative Volatility Research Engine")
st.caption("Walk-forward ML + VIX + observed option-chain analytics")

with st.sidebar:
    ticker = st.text_input("Ticker", "SPY").upper()
    start = st.date_input("Historical start", pd.Timestamp("2010-01-01"))
    threshold = st.slider(
        "High-volatility probability threshold",
        0.40, 0.80, 0.55, 0.01
    )

@st.cache_data(ttl=3600)
def prepare(ticker, start):
    prices = download_history(ticker, str(start))
    vix = download_vix(str(start))
    data = build_features(prices, vix)
    return prices, add_forward_regime(data)

try:
    prices, data = prepare(ticker, start)
    predictions = walk_forward_predict(data, FEATURE_COLUMNS)
except Exception as exc:
    st.error(str(exc))
    st.stop()

metrics = evaluate(predictions)
bt = run_regime_backtest(prices, predictions, threshold)
stats = performance_stats(bt)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Walk-forward accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
c3.metric("Strategy Sharpe", f"{stats.get('sharpe', float('nan')):.2f}")
c4.metric("Max drawdown", f"{stats.get('max_drawdown', float('nan')):.1%}")

st.subheader("VIX and realized volatility")
vol = data.reset_index()
date_col = vol.columns[0]
fig = px.line(
    vol,
    x=date_col,
    y=["rv_20", "vix_level"],
    title="20-day realized volatility vs VIX",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Random Forest probabilities")
prob = predictions.reset_index()
date_col = prob.columns[0]
fig_prob = px.area(
    prob,
    x=date_col,
    y=["prob_Low", "prob_Medium", "prob_High"],
    title="Walk-forward regime probabilities",
)
st.plotly_chart(fig_prob, use_container_width=True)

st.subheader("Feature importance")
imp = final_model_importance(data, FEATURE_COLUMNS).sort_values().reset_index()
imp.columns = ["feature", "importance"]
fig_imp = px.bar(imp, x="importance", y="feature", orientation="h")
st.plotly_chart(fig_imp, use_container_width=True)

st.subheader("Research backtest")
bt_plot = bt.reset_index()
date_col = bt_plot.columns[0]
fig_bt = px.line(
    bt_plot,
    x=date_col,
    y=["buy_hold_curve", "strategy_curve"],
    title="Cumulative research curves",
)
st.plotly_chart(fig_bt, use_container_width=True)

st.subheader("Current observed option chain")

try:
    chain = current_option_chain(ticker)
    spot = float(prices["close"].iloc[-1])
    cleaned = clean_option_chain(chain, spot)

    if cleaned.empty:
        st.warning("No usable option contracts after filtering.")
    else:
        st.write(
            f"Loaded {len(cleaned):,} listed contracts with observed implied volatility."
        )
        surface = option_surface(cleaned)

        fig_surface = px.scatter(
            surface,
            x="strike",
            y="days_to_expiry",
            size="openInterest",
            color="impliedVolatility",
            hover_data=["option_type", "volume"],
            title="Observed current option implied-volatility surface",
        )
        st.plotly_chart(fig_surface, use_container_width=True)

        st.dataframe(
            cleaned[
                [
                    "expiration",
                    "option_type",
                    "strike",
                    "bid",
                    "ask",
                    "volume",
                    "openInterest",
                    "impliedVolatility",
                ]
            ].sort_values(["expiration", "strike"])
        )

except Exception as exc:
    st.info(
        "Current option-chain data were not available for this ticker in the "
        f"data source: {exc}"
    )

with st.expander("Classification report"):
    st.text(metrics["classification_report"])

st.caption(
    "Research only. The backtest is deliberately simplified and excludes "
    "transaction costs, slippage, financing and execution constraints."
)
