# training_conversion.py
# Builds labeled training_data.csv from vehicle_positions_log.csv using a distance threshold "arrival" rule,
# and adds progress_rate (meters per second getting closer to stop).

import pandas as pd

LOG = "vehicle_positions_log.csv"
OUT = "training_data.csv"

THRESHOLD_M = 50          # try 30, 50, 75
MAX_ETA_SEC = 20 * 60     # discard crazy labels

df = pd.read_csv(LOG)

# Keep only rows that can be labeled
df = df.dropna(subset=[
    "vehicle_id", "trip_id", "route_id", "stop_id",
    "feed_timestamp", "dist_to_stop_m", "speed_mps"
]).copy()

df = df.sort_values(["vehicle_id", "trip_id", "feed_timestamp"])

# Create "segments" where stop_id stays the same for a vehicle+trip
stop_changed = df["stop_id"] != df.groupby(["vehicle_id", "trip_id"])["stop_id"].shift(1)
df["segment_id"] = stop_changed.groupby([df["vehicle_id"], df["trip_id"]]).cumsum()

labeled_parts = []

for (veh, trip, seg), g in df.groupby(["vehicle_id", "trip_id", "segment_id"], sort=False):
    # Find first time it gets close enough to the stop
    close = g[g["dist_to_stop_m"] <= THRESHOLD_M]
    if close.empty:
        continue

    arrival_ts = close["feed_timestamp"].iloc[0]

    gg = g.copy()
    gg["eta_seconds"] = arrival_ts - gg["feed_timestamp"]

    # Keep only rows before arrival (eta>0) and within reasonable range
    gg = gg[(gg["eta_seconds"] > 0) & (gg["eta_seconds"] <= MAX_ETA_SEC)]

    if not gg.empty:
        labeled_parts.append(gg)

train = pd.concat(labeled_parts, ignore_index=True) if labeled_parts else pd.DataFrame()

if not train.empty:
    train = train.sort_values(["vehicle_id", "trip_id", "feed_timestamp"]).copy()

    # --- Time features ---
    dt = pd.to_datetime(train["feed_timestamp"], unit="s", utc=True)
    train["hour"] = dt.dt.hour
    train["dow"] = dt.dt.dayofweek
    train["is_weekend"] = (train["dow"] >= 5).astype(int)
    train["stopped_flag"] = (train["speed_mps"] < 0.5).astype(int)

    # --- NEW: progress_rate feature (meters per second getting closer to stop) ---
    train["prev_dist"] = train.groupby(["vehicle_id", "trip_id"])["dist_to_stop_m"].shift(1)
    train["prev_ts"] = train.groupby(["vehicle_id", "trip_id"])["feed_timestamp"].shift(1)

    dt_sec = (train["feed_timestamp"] - train["prev_ts"]).clip(lower=1)
    train["progress_rate"] = (train["prev_dist"] - train["dist_to_stop_m"]) / dt_sec
    train["progress_rate"] = train["progress_rate"].fillna(0)

    # Cleanup helper columns
    train = train.drop(columns=["prev_dist", "prev_ts"], errors="ignore")

train.to_csv(OUT, index=False)
print("Wrote", OUT, "rows:", len(train))

if not train.empty:
    print(train["eta_seconds"].describe())
    print(train["eta_seconds"].value_counts().head(10))
    print("\nprogress_rate summary:")
    print(train["progress_rate"].describe())
