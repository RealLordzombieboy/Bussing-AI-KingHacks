import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gtfs_schedule import load_stop_times_lookup
from tripupdates_parse import parse_tripupdates_pb

GTFS_ZIP = "gtfs.zip"
VEH_LOG = "vehicle_positions_log.csv"
TRIPUPDATES_PB = "tripupdates.pb"   # or a freshly downloaded snapshot
MODEL_PATH = "eta_model.joblib"

TZ = ZoneInfo("America/Toronto")

def yyyymmdd_to_date(s: str):
    return datetime.strptime(str(s), "%Y%m%d").date()

def scheduled_unix_from_start_date(start_date_yyyymmdd: str, sched_arrival_sec: int) -> int:
    """
    GTFS times can be > 24h. If sched_arrival_sec = 25:10:00, it means next day.
    start_date is local service date.
    """
    base = datetime.combine(yyyymmdd_to_date(start_date_yyyymmdd), datetime.min.time()).replace(tzinfo=TZ)
    day_offset = sched_arrival_sec // 86400
    sec_in_day = sched_arrival_sec % 86400
    dt_local = base + timedelta(days=day_offset, seconds=sec_in_day)
    # convert to unix UTC
    return int(dt_local.astimezone(ZoneInfo("UTC")).timestamp())

def add_time_features(df):
    dt = pd.to_datetime(df["feed_timestamp"], unit="s", utc=True)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["stopped_flag"] = (df["speed_mps"] < 0.5).astype(int)
    return df

def main():
    # 1) latest snapshot from your vehicle log
    vdf = pd.read_csv(VEH_LOG)
    latest_ts = vdf["feed_timestamp"].max()
    snap = vdf[vdf["feed_timestamp"] == latest_ts].copy()

    # require the join keys + model essentials
    snap = snap.dropna(subset=["trip_id", "route_id", "stop_id", "current_stop_sequence", "dist_to_stop_m", "speed_mps"])
    snap["trip_id"] = snap["trip_id"].astype(str)
    snap["route_id"] = snap["route_id"].astype(str)
    snap["stop_id"] = snap["stop_id"].astype(str)
    snap["current_stop_sequence"] = pd.to_numeric(snap["current_stop_sequence"], errors="coerce").astype("Int64")
    snap = snap.dropna(subset=["current_stop_sequence"])

    # If you added trip_start_date in data_conversion2.py, this will be present.
    # If not present, we’ll approximate with local date from feed_timestamp (works most of the time).
    if "trip_start_date" not in snap.columns or snap["trip_start_date"].isna().all():
        local_date = datetime.fromtimestamp(int(latest_ts), tz=ZoneInfo("UTC")).astimezone(TZ).strftime("%Y%m%d")
        snap["trip_start_date"] = local_date

    # 2) scheduled lookup from GTFS static
    st = load_stop_times_lookup(GTFS_ZIP)
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_sequence"] = st["stop_sequence"].astype(int)

    snap = snap.merge(
        st,
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left"
    )

    snap = snap.dropna(subset=["sched_arrival_sec"]).copy()

    # convert scheduled to unix
    snap["sched_arrival_ts"] = snap.apply(
        lambda r: scheduled_unix_from_start_date(r["trip_start_date"], int(r["sched_arrival_sec"])),
        axis=1
    )

    # 3) parse tripupdates predictions
    tu = parse_tripupdates_pb(TRIPUPDATES_PB)
    tu = tu.dropna(subset=["rt_pred_arrival_ts"])
    tu["stop_sequence"] = tu["stop_sequence"].astype(int)

    snap = snap.merge(
        tu[["trip_id", "stop_sequence", "rt_pred_arrival_ts"]],
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left"
    )

    # 4) model prediction (AI ETA)
    pipe = joblib.load(MODEL_PATH)

    snap = add_time_features(snap)

    # If you’re not logging these in snap, set to 0 (safe)
    for col in ["progress_rate", "progress_rate_ewm", "speed_ewm"]:
        if col not in snap.columns:
            snap[col] = 0.0

    num_features = [
        "dist_to_stop_m","speed_mps","speed_ewm","progress_rate","progress_rate_ewm",
        "bearing","current_stop_sequence","congestion_level","hour","dow","is_weekend","stopped_flag"
    ]
    cat_features = ["route_id","stop_id"]

    snap[num_features] = snap[num_features].fillna(0)

    eta_pred_s = pipe.predict(snap[num_features + cat_features])
    snap["ai_pred_arrival_ts"] = snap["feed_timestamp"] + np.maximum(eta_pred_s, 0)

    # 5) compute delays (minutes)
    snap["sched_arrival_local"] = pd.to_datetime(snap["sched_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)
    snap["ai_arrival_local"] = pd.to_datetime(snap["ai_pred_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)

    snap["ai_delay_min"] = (snap["ai_pred_arrival_ts"] - snap["sched_arrival_ts"]) / 60.0
    snap["rt_delay_min"] = (snap["rt_pred_arrival_ts"] - snap["sched_arrival_ts"]) / 60.0

    # show a clean table (soonest scheduled arrivals or biggest delays)
    out = snap[[
        "route_id","stop_id","stop_name","current_stop_sequence",
        "sched_arrival_local","ai_arrival_local","ai_delay_min",
        "rt_pred_arrival_ts","rt_delay_min"
    ]].copy()

    # Pretty RT time column
    out["rt_arrival_local"] = pd.to_datetime(out["rt_pred_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)
    out = out.drop(columns=["rt_pred_arrival_ts"])

    out = out.sort_values("sched_arrival_local").head(15)

    print("\n=== Delay Comparison (Schedule vs Official RT vs Our AI) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
