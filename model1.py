# model1.py
# Trains a first ETA model from training_data.csv and compares it to a simple baseline.
# Uses progress_rate (how quickly distance-to-stop is shrinking) if present.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = "training_data.csv"


def main():
    df = pd.read_csv(DATA)

    # Keep only rows with the essentials
    required = ["eta_seconds", "dist_to_stop_m", "speed_mps", "route_id", "stop_id", "feed_timestamp"]
    df = df.dropna(subset=required)

    # Ensure categorical columns are strings (avoids weird numeric parsing)
    df["route_id"] = df["route_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    # Baseline: distance / speed (avoid divide-by-zero)
    df["eta_baseline"] = df["dist_to_stop_m"] / np.maximum(df["speed_mps"], 1.0)

    # Train/test split for first test (later you can do time-based split)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    y_train = train_df["eta_seconds"].values
    y_test = test_df["eta_seconds"].values

    # Features
    num_features = [
        "dist_to_stop_m",
        "speed_mps",
        "progress_rate",          # <-- NEW
        "bearing",
        "current_stop_sequence",
        "congestion_level",
        "hour",
        "dow",
        "is_weekend",
        "stopped_flag",
    ]
    cat_features = ["route_id", "stop_id"]

    # If progress_rate isn't in the data yet, create it as 0
    if "progress_rate" not in df.columns:
        train_df["progress_rate"] = 0.0
        test_df["progress_rate"] = 0.0

    # Fill missing numeric values
    train_df[num_features] = train_df[num_features].fillna(0)
    test_df[num_features] = test_df[num_features].fillna(0)

    # IMPORTANT: HistGradientBoostingRegressor expects dense input.
    # Make OneHotEncoder output dense by setting sparse_output=False (sklearn>=1.2).
    # If your sklearn is older and this errors, change to sparse=False.
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train = train_df[num_features + cat_features]
    X_test = test_df[num_features + cat_features]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae_model = mean_absolute_error(y_test, pred)
    mae_baseline = mean_absolute_error(y_test, test_df["eta_baseline"].values)

    print(f"Baseline MAE: {mae_baseline/60:.2f} minutes")
    print(f"Model MAE:    {mae_model/60:.2f} minutes")

    # Show a few predictions for sanity-checking
    out_cols = ["route_id", "stop_id", "dist_to_stop_m", "speed_mps", "progress_rate", "eta_seconds", "eta_baseline"]
    out = test_df[out_cols].copy()
    out["eta_pred"] = pred
    out["eta_pred_min"] = out["eta_pred"] / 60
    out["eta_true_min"] = out["eta_seconds"] / 60

    print("\nSample predictions:")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
