# model1.py
"""
Train an ETA-ish model from snapshot CSVs that look like:
trip_id,route_id,stop_id,speed_mps,feed_timestamp,vehicle_id
... where feed_timestamp is usually HH:MM:SS (time-of-day).

LABEL (proxy):
  eta_seconds = seconds until this vehicle's stop_id changes (next stop transition)
This is not "arrival time at stop" from GTFS — it's a usable proxy when you only
have snapshots.

USAGE:
  python model1.py --data_dir "Jan 15 data"
  python model1.py --glob "Jan 15 data/*.csv"
  python model1.py --glob "/path/to/current_data*.csv"

OUTPUT:
  eta_model.joblib
"""

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


MODEL_PATH = "eta_model.joblib"


def norm_stop_id(s) -> Optional[str]:
    """Normalize stop IDs so '00278' / '278' / 'S278' become 'S00278'."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    # pure digits -> prefix S + zfill
    if re.fullmatch(r"\d+", s):
        return "S" + s.zfill(5)
    # S#### or s#### -> normalize
    m = re.fullmatch(r"[Ss](\d+)", s)
    if m:
        return "S" + m.group(1).zfill(5)
    return s


def clean_id(x) -> Optional[str]:
    """Turn 501.0 -> '501', keep '19A' as '19A', None -> None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def parse_feed_time_to_seconds(x) -> float:
    """
    Accept:
      - HH:MM:SS (or HH:MM)  -> seconds since midnight
      - unix epoch (int/float-like) -> return as epoch seconds
    For snapshot datasets with HH:MM:SS, we'll use seconds-since-midnight ordering.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    # epoch-like?
    if re.fullmatch(r"\d{9,12}(\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return np.nan

    # HH:MM:SS or HH:MM
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        return float(hh * 3600 + mm * 60 + ss)

    return np.nan


def load_many_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["_source_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {repr(e)}")

    if not dfs:
        raise RuntimeError("No CSVs could be loaded. Check your --glob/--data_dir path.")
    return pd.concat(dfs, ignore_index=True)


def pick_paths(args) -> List[str]:
    if args.glob:
        paths = sorted(glob.glob(args.glob))
    else:
        data_dir = args.data_dir
        patterns = [os.path.join(data_dir, "*.csv"), os.path.join(data_dir, "**/*.csv")]
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(pat, recursive=True))
        paths = sorted(set(paths))

    # optionally filter
    if args.name_contains:
        paths = [p for p in paths if args.name_contains in os.path.basename(p)]

    return paths


def label_eta_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy labels:
      For each vehicle_id group, sort by time_sec, create segments where stop_id changes.
      For each segment, next_segment_first_time is when the vehicle hits a new stop_id.
      For each row in the segment:
         eta_seconds = next_segment_first_time - current_time_sec
    """
    df = df.copy()

    # group sort key
    df = df.sort_values(["vehicle_id", "time_sec"], kind="mergesort")

    def label_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("time_sec", kind="mergesort").copy()
        # segment id increments when stop_id changes
        g["seg"] = (g["stop_id"] != g["stop_id"].shift()).cumsum()

        seg_first = g.groupby("seg")["time_sec"].first()
        seg_next_first = seg_first.shift(-1)  # next segment's first time
        mapping = seg_next_first.to_dict()

        g["next_stop_time_sec"] = g["seg"].map(mapping)
        g["eta_seconds"] = g["next_stop_time_sec"] - g["time_sec"]

        return g

    df = df.groupby("vehicle_id", group_keys=False).apply(label_group)

    # keep reasonable labels
    df = df.dropna(subset=["eta_seconds"]).copy()
    df = df[(df["eta_seconds"] > 0) & (df["eta_seconds"] <= 60 * 30)].copy()  # <= 30 min cap

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    If time_sec is seconds since midnight: derive hour/minute.
    If time_sec is epoch: still derive hour/minute in UTC-ish by modulo.
    """
    df = df.copy()
    sec = pd.to_numeric(df["time_sec"], errors="coerce")
    sec_mod = (sec % 86400).fillna(0)
    df["hour"] = (sec_mod // 3600).astype(int)
    df["minute"] = ((sec_mod % 3600) // 60).astype(int)
    df["stopped_flag"] = (pd.to_numeric(df["speed_mps"], errors="coerce").fillna(0) < 0.5).astype(int)
    return df


def train_model(train_df: pd.DataFrame) -> Pipeline:
    y = train_df["eta_seconds"].astype(float).values

    # features
    X = train_df[[
        "speed_mps",
        "hour",
        "minute",
        "stopped_flag",
        "route_id",
        "stop_id",
    ]].copy()

    # preprocess
    numeric = ["speed_mps", "hour", "minute", "stopped_flag"]
    cat = ["route_id", "stop_id"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5), cat),
        ],
        remainder="drop",
    )

    # model
    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(X, y)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="Jan 15 data", help="Directory containing snapshot CSVs")
    ap.add_argument("--glob", default="", help="Glob pattern for CSVs (overrides --data_dir)")
    ap.add_argument("--name_contains", default="", help="Only use CSV files whose filename contains this text")
    args = ap.parse_args()

    paths = pick_paths(args)
    if not paths:
        raise RuntimeError("No CSV files found. Check your paths and flags.")

    print(f"Found {len(paths)} CSV(s). Loading…")
    df = load_many_csvs(paths)

    # expected columns
    needed = {"trip_id", "route_id", "stop_id", "speed_mps", "feed_timestamp", "vehicle_id"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Columns found: {list(df.columns)}")

    # clean / normalize
    df["trip_id"] = df["trip_id"].apply(clean_id).astype(str)
    df["route_id"] = df["route_id"].apply(clean_id).astype(str)
    df["stop_id"] = df["stop_id"].apply(norm_stop_id)
    df["vehicle_id"] = df["vehicle_id"].apply(clean_id).astype(str)
    df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")

    df["time_sec"] = df["feed_timestamp"].apply(parse_feed_time_to_seconds)

    # drop unusable
    df = df.dropna(subset=["vehicle_id", "route_id", "stop_id", "speed_mps", "time_sec"]).copy()

    print(f"Loaded rows total (usable after cleaning): {len(df):,}")

    # label
    labeled = label_eta_seconds(df)
    print(f"Training rows after labeling: {len(labeled):,}")

    if labeled.empty:
        print("\n[WHY THIS HAPPENS]")
        print("No labeled rows means we never observed a vehicle's stop_id change over time.")
        print("Common causes:")
        print("  • You only loaded ONE snapshot CSV (need many over time).")
        print("  • stop_id doesn't change in your data for each vehicle.")
        print("  • feed_timestamp isn't actually a time you can sort by.")
        print("\nTry:")
        print("  python model1.py --glob 'Jan 15 data/*.csv'")
        return

    # time features
    labeled = add_time_features(labeled)

    # summary
    print("\nETA label summary (seconds):")
    print(labeled["eta_seconds"].describe())

    # split by vehicle
    X_all = labeled
    y_all = labeled["eta_seconds"].astype(float).values
    groups = labeled["vehicle_id"].astype(str).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups=groups))

    train_df = labeled.iloc[train_idx].copy()
    test_df = labeled.iloc[test_idx].copy()

    pipe = train_model(train_df)

    # eval
    X_test = test_df[["speed_mps", "hour", "minute", "stopped_flag", "route_id", "stop_id"]].copy()
    y_test = test_df["eta_seconds"].astype(float).values
    pred = pipe.predict(X_test)
    pred = np.maximum(pred, 0)

    # baseline = median label from train
    baseline = np.median(train_df["eta_seconds"].astype(float).values)
    baseline_pred = np.full_like(y_test, baseline, dtype=float)

    mae_model = mean_absolute_error(y_test, pred)
    mae_base = mean_absolute_error(y_test, baseline_pred)

    print("\nModel evaluation (proxy ETA to next stop change):")
    print(f"Baseline MAE: {mae_base/60:.2f} minutes")
    print(f"Model MAE:    {mae_model/60:.2f} minutes")

    # show sample predictions
    sample = test_df[[
        "route_id", "stop_id", "vehicle_id", "speed_mps", "hour", "minute", "eta_seconds"
    ]].copy()
    sample["eta_pred_s"] = pred
    sample["eta_true_min"] = sample["eta_seconds"] / 60.0
    sample["eta_pred_min"] = sample["eta_pred_s"] / 60.0

    print("\nSample predictions:")
    print(sample.head(15).to_string(index=False))

    # save
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
