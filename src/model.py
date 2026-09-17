from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)


REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def make_model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def walk_forward_predict(
    data: pd.DataFrame,
    feature_columns: list[str],
    min_train=750,
    test_block=63,
):
    X = data[feature_columns]
    y = data["regime"].astype(int)

    predictions = []
    probabilities = []
    indices = []

    for train_end in range(min_train, len(data), test_block):
        test_end = min(train_end + test_block, len(data))

        model = make_model()
        model.fit(X.iloc[:train_end], y.iloc[:train_end])

        p = model.predict(X.iloc[train_end:test_end])
        proba = model.predict_proba(X.iloc[train_end:test_end])

        predictions.extend(p)
        probabilities.extend(proba)
        indices.extend(data.index[train_end:test_end])

    result = data.loc[indices, ["close", "forward_vol", "regime"]].copy()
    result["predicted_regime"] = predictions

    prob = pd.DataFrame(
        probabilities,
        index=result.index,
        columns=[f"prob_{REGIME_NAMES[c]}" for c in range(3)],
    )
    result = result.join(prob)

    return result


def final_model_importance(data, feature_columns):
    model = make_model()
    model.fit(data[feature_columns], data["regime"].astype(int))
    return pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)


def evaluate(result):
    y = result["regime"]
    p = result["predicted_regime"]

    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "classification_report": classification_report(
            y,
            p,
            target_names=["Low", "Medium", "High"],
            zero_division=0,
        ),
    }
