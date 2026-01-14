# model1.py
# Trains an ETA model from training_data.csv and compares it to a baseline.
# Improvements:
#  - time-based split (more realistic)
#  - improved baseline (uses 0.5 m/s floor + cap)
#  - uses smoothed features: progress_rate_ewm, speed_ewm
#  - prints extra diagnostics including long-ETA MAE

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = "training_data.csv"


def main():
    df = pd.read_csv(DATA)

    required = ["eta_seconds", "dist_to_stop_m", "speed_mps", "route_id", "stop_id", "feed_timestamp"]
    df = df.dropna(subset=required).copy()

    # Ensure categorical columns are strings
    df["route_id"] = df["route_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    # Improved baseline: distance / max(speed, 0.5), capped to 20 minutes
    df["eta_baseline"] = df["dist_to_stop_m"] / np.maximum(df["speed_mps"], 0.5)
    df["eta_baseline"] = np.minimum(df["eta_baseline"], 20 * 60)

    # Sort + time-based split
    df = df.sort_values("feed_timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.75)
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()

    y_train = train_df["eta_seconds"].values
    y_test = test_df["eta_seconds"].values

    # Feature list (include smoothed features)
    num_features = [
        "dist_to_stop_m",
        "speed_mps",
        "speed_ewm",
        "progress_rate",
        "progress_rate_ewm",
        "bearing",
        "current_stop_sequence",
        "congestion_level",
        "hour",
        "dow",
        "is_weekend",
        "stopped_flag",
    ]
    cat_features = ["route_id", "stop_id"]

    # If any new columns are missing (older training file), create them
    for col in ["speed_ewm", "progress_rate", "progress_rate_ewm"]:
        if col not in df.columns:
            train_df[col] = 0.0
            test_df[col] = 0.0

    # Fill missing numeric
    train_df[num_features] = train_df[num_features].fillna(0)
    test_df[num_features] = test_df[num_features].fillna(0)

    # Dense one-hot encoding for HistGradientBoostingRegressor
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        max_depth=7,
        learning_rate=0.06,
        max_iter=600,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train = train_df[num_features + cat_features]
    X_test = test_df[num_features + cat_features]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae_model = mean_absolute_error(y_test, pred)
    mae_baseline = mean_absolute_error(y_test, test_df["eta_baseline"].values)

    print("Rows:", len(df), "| Train:", len(train_df), "| Test:", len(test_df))
    print("Unique routes:", df["route_id"].nunique(), "| Unique stops:", df["stop_id"].nunique())
    print("\nETA counts (top):")
    print(df["eta_seconds"].value_counts().head(10))

    print(f"\nBaseline MAE: {mae_baseline/60:.2f} minutes")
    print(f"Model MAE:    {mae_model/60:.2f} minutes")

    # Long ETA evaluation (>= 2 minutes)
    mask_long = y_test >= 120
    if mask_long.any():
        mae_baseline_long = mean_absolute_error(y_test[mask_long], test_df.loc[mask_long, "eta_baseline"].values)
        mae_model_long = mean_absolute_error(y_test[mask_long], pred[mask_long])
        print("\nMAE on long ETAs (>= 2 min):")
        print(f"Baseline: {mae_baseline_long/60:.2f} minutes")
        print(f"Model:    {mae_model_long/60:.2f} minutes")
    else:
        print("\nNo long-ETA examples (>= 2 min) in test split this run.")

    # Sample predictions
    out_cols = [
        "route_id", "stop_id", "dist_to_stop_m", "speed_mps",
        "speed_ewm", "progress_rate", "progress_rate_ewm",
        "eta_seconds", "eta_baseline"
    ]
    out = test_df[out_cols].copy()
    out["eta_pred"] = pred
    out["eta_pred_min"] = out["eta_pred"] / 60
    out["eta_true_min"] = out["eta_seconds"] / 60

    print("\nSample predictions:")
    print(out.head(12).to_string(index=False))

    print("\nPrediction summary:")
    print(pd.Series(pred).describe())


if __name__ == "__main__":
    main()
