from __future__ import annotations

import argparse

from .backtest import performance_stats, run_regime_backtest
from .data import download_history, download_vix
from .features import FEATURE_COLUMNS, add_forward_regime, build_features
from .model import evaluate, final_model_importance, walk_forward_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2010-01-01")
    args = parser.parse_args()

    prices = download_history(args.ticker, args.start)
    vix = download_vix(args.start)

    data = build_features(prices, vix)
    data = add_forward_regime(data)

    predictions = walk_forward_predict(
        data,
        FEATURE_COLUMNS,
        min_train=750,
        test_block=63,
    )

    metrics = evaluate(predictions)
    print("\n=== WALK-FORWARD MODEL ===")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.3f}")
    print(metrics["classification_report"])

    print("\n=== FEATURE IMPORTANCE ===")
    print(final_model_importance(data, FEATURE_COLUMNS).to_string())

    bt = run_regime_backtest(prices, predictions)
    stats = performance_stats(bt)

    print("\n=== SIMPLE REGIME BACKTEST ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    predictions.to_csv("walk_forward_predictions.csv")
    bt.to_csv("regime_backtest.csv")


if __name__ == "__main__":
    main()
