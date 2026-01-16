# training_conversion.py
# Builds labeled training_data.csv from vehicle_positions_log.csv using a distance threshold "arrival" rule.
# Improvements:
#  - progress_rate computed within (vehicle_id, trip_id, segment_id)
#  - arrival rule reduces GPS noise: (dist<=THRESHOLD) AND (speed<=1.0 OR dist<=25m)
#  - adds smoothed features: progress_rate_ewm, speed_ewm

import pandas as pd

LOG = "vehicle_positions_log.csv"
OUT = "training_data.csv"

THRESHOLD_M = 60           # was 50; slightly higher to find arrival more often
MAX_ETA_SEC = 20 * 60      # discard crazy labels


def main():
    df = pd.read_csv(LOG)

    # Keep only rows that can be labeled
    df = df.dropna(subset=[
        "vehicle_id", "trip_id", "route_id", "stop_id",
        "feed_timestamp", "dist_to_stop_m", "speed_mps"
    ]).copy()

    df = df.sort_values(["vehicle_id", "trip_id", "feed_timestamp"])

    # Segments where stop_id stays constant for a vehicle+trip
    stop_changed = df["stop_id"] != df.groupby(["vehicle_id", "trip_id"])["stop_id"].shift(1)
    df["segment_id"] = stop_changed.groupby([df["vehicle_id"], df["trip_id"]]).cumsum()

    # progress_rate within segment
    df["prev_dist"] = df.groupby(["vehicle_id", "trip_id", "segment_id"])["dist_to_stop_m"].shift(1)
    df["prev_ts"] = df.groupby(["vehicle_id", "trip_id", "segment_id"])["feed_timestamp"].shift(1)
    dt_sec = (df["feed_timestamp"] - df["prev_ts"]).clip(lower=1)
    df["progress_rate"] = (df["prev_dist"] - df["dist_to_stop_m"]) / dt_sec
    df["progress_rate"] = df["progress_rate"].fillna(0)
    df = df.drop(columns=["prev_dist", "prev_ts"], errors="ignore")

    labeled_parts = []
    segments_total = 0
    segments_labeled = 0

    for (veh, trip, seg), g in df.groupby(["vehicle_id", "trip_id", "segment_id"], sort=False):
        segments_total += 1

        # Improved arrival condition (reduces noisy "teleporting" close points):
        # - within THRESHOLD_M AND either slow/stopped OR extremely close (<=25m)
        close = g[
            (g["dist_to_stop_m"] <= THRESHOLD_M) &
            ((g["speed_mps"] <= 1.0) | (g["dist_to_stop_m"] <= 25.0))
        ]

        if close.empty:
            continue

        segments_labeled += 1
        arrival_ts = close["feed_timestamp"].iloc[0]

        gg = g.copy()
        gg["eta_seconds"] = arrival_ts - gg["feed_timestamp"]

        # Keep only rows before arrival (eta>0) and within reasonable range
        gg = gg[(gg["eta_seconds"] > 0) & (gg["eta_seconds"] <= MAX_ETA_SEC)]

        if not gg.empty:
            labeled_parts.append(gg)

    train = pd.concat(labeled_parts, ignore_index=True) if labeled_parts else pd.DataFrame()

    if train.empty:
        train.to_csv(OUT, index=False)
        print(f"Wrote {OUT} rows: 0")
        print(f"Segments total: {segments_total} | segments labeled: {segments_labeled} (threshold={THRESHOLD_M}m)")
        return

    # Sort for time features + smoothing
    train = train.sort_values(["vehicle_id", "trip_id", "segment_id", "feed_timestamp"]).copy()

    # Time features
    dt = pd.to_datetime(train["feed_timestamp"], unit="s", utc=True)
    train["hour"] = dt.dt.hour
    train["dow"] = dt.dt.dayofweek
    train["is_weekend"] = (train["dow"] >= 5).astype(int)
    train["stopped_flag"] = (train["speed_mps"] < 0.5).astype(int)

    # Smoothed features within segment
    train["progress_rate_ewm"] = (
        train.groupby(["vehicle_id", "trip_id", "segment_id"])["progress_rate"]
        .transform(lambda s: s.ewm(span=3, adjust=False).mean())
    )
    train["speed_ewm"] = (
        train.groupby(["vehicle_id", "trip_id", "segment_id"])["speed_mps"]
        .transform(lambda s: s.ewm(span=3, adjust=False).mean())
    )

    train.to_csv(OUT, index=False)

    # Checks
    print(f"Wrote {OUT} rows: {len(train)}")
    print(f"Segments total: {segments_total} | segments labeled: {segments_labeled} (threshold={THRESHOLD_M}m)")

    print("\neta_seconds distribution:")
    print(train["eta_seconds"].describe())
    print(train["eta_seconds"].value_counts().head(10))

    print("\nprogress_rate summary:")
    print(train["progress_rate"].describe())

    print("\nprogress_rate_ewm summary:")
    print(train["progress_rate_ewm"].describe())

    print("\nspeed_ewm summary:")
    print(train["speed_ewm"].describe())


if __name__ == "__main__":
    main()
