# Quantitative Volatility Research Engine

**Market regime detection, volatility forecasting, options analytics and walk-forward machine learning**

A research-oriented Python project designed to study the interaction between **market returns, volatility, VIX, option-implied volatility and machine learning**.

The central research question is:

> Can observable market and volatility information improve the classification of the next volatility regime?

The project deliberately separates **data engineering, feature construction, model validation, options analytics and signal research**.

## Why this version uses real option data

When available, the pipeline uses **actual option-chain snapshots** rather than inventing option prices.

Free sources can provide current option chains for selected underlyings, while historical option-chain datasets are often licensed/paywalled. The repository therefore has two modes:

1. **Live/current option chain mode**, retrieves real listed contracts from Yahoo Finance when available.
2. **Historical research mode**, trains the model primarily on underlying + VIX history, while the option surface module operates on actual current-chain snapshots.

This distinction is intentional: the project never presents synthetic options as if they were historical market observations.

## Research pipeline

```text
Underlying prices ─┐
                   ├──> Feature Engineering ──> Walk-Forward RF ──> Probabilities
VIX ───────────────┤                                      │
                   │                                      └──> Regime Signal
Option Chain ──────┴──> IV Surface / Greeks

Regime Signal ──> Simple Backtest ──> Research Diagnostics
```

## Main components

### 1. Market data

- SPY / QQQ / IWM or another Yahoo Finance ticker
- VIX (`^VIX`)
- OHLCV
- daily returns

### 2. Volatility features

- 5 / 20 / 60-day realized volatility
- volatility ratio
- return momentum
- moving-average distance
- drawdown
- volume z-score
- VIX level
- VIX change
- VIX term-independent momentum

### 3. Machine learning

Random Forest classifier with:

- chronological train/test split
- expanding walk-forward validation
- probability predictions
- feature importance
- accuracy / balanced accuracy
- classification report
- confusion matrix

### 4. Options

Current listed option chains are pulled from Yahoo Finance when accessible.

For each available contract the pipeline extracts:

- strike
- expiration
- call/put
- last price
- bid / ask
- volume
- open interest
- implied volatility

It then constructs an IV surface from **observed market IV** and calculates Black-Scholes Greeks for selected contracts.

### 5. Backtest

The research signal uses the predicted probability of the high-volatility regime.

Example rule:

```text
P(high volatility) > threshold
        ↓
reduce / avoid long-risk exposure
```

The backtest is deliberately simple. It is a research diagnostic, not a production trading strategy.

## Important data limitation

Historical option chains are the main data bottleneck for an independent project.

The project does **not** fabricate historical option observations to hide this limitation.

Instead:

- historical ML research uses freely available underlying/VIX history;
- current option analytics use actual listed option-chain observations;
- historical option-chain research can later be plugged into the same interface if a licensed dataset is available.

This is preferable to claiming that a synthetic surface represents the actual market.

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Run research

```bash
python -m src.research --ticker SPY
```

## Run dashboard

```bash
streamlit run dashboard/app.py
```

## Project structure

```text
quant-volatility-engine/
├── data/
├── dashboard/
│   └── app.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── options.py
│   ├── model.py
│   ├── backtest.py
│   └── research.py
├── requirements.txt
└── README.md
```

## Why walk-forward validation?

Financial observations are ordered in time. A random train/test shuffle can allow information from later periods to influence model selection.

The walk-forward procedure instead repeatedly:

1. trains on the past;
2. predicts the next block;
3. moves the training window forward;
4. evaluates only observations that occur after the training sample.

## Why probabilities matter

A classifier output such as:

```text
Low = 0.08
Medium = 0.22
High = 0.70
```

contains more information than simply saying `High`.

The dashboard therefore exposes the full class probabilities. The research signal can then use a configurable probability threshold.

## Disclaimer

This repository is for quantitative research and education. It does not constitute investment advice and does not claim a profitable trading strategy.
