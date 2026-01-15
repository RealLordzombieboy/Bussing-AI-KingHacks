import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gtfs_schedule import load_stop_times_lookup
from tripupdates_parse import parse_tripupdates_pb

# ====== FILES ======
GTFS_ZIP = "gtfs.zip"
VEH_LOG = "vehicle_positions_log.csv"
TRIPUPDATES_INDEX = "tripupdates_index.csv"   # produced by your capture script
MODEL_PATH = "eta_model.joblib"

OUT_HTML = "delay_comparison_latest.html"
OUT_CSV = "delay_comparison_latest.csv"

TZ = ZoneInfo("America/Toronto")
UTC = ZoneInfo("UTC")


# ---------- helpers ----------
def yyyymmdd_to_date(s: str):
    s = str(s).strip()
    # handle "20260114.0" or floats read from CSV
    s = s.replace(".0", "")
    return datetime.strptime(s, "%Y%m%d").date()


def scheduled_unix_from_start_date(start_date_yyyymmdd: str, sched_arrival_sec: int) -> int:
    """
    GTFS times can be > 24h. If sched_arrival_sec=25:10:00, it means next day.
    start_date is local service date (America/Toronto).
    """
    base = datetime.combine(yyyymmdd_to_date(start_date_yyyymmdd), datetime.min.time()).replace(tzinfo=TZ)
    day_offset = sched_arrival_sec // 86400
    sec_in_day = sched_arrival_sec % 86400
    dt_local = base + timedelta(days=day_offset, seconds=sec_in_day)
    return int(dt_local.astimezone(UTC).timestamp())


def norm_stop_id(s) -> str:
    if pd.isna(s):
        return None
    s = str(s).strip()
    # if it looks like "278" or "00278" -> S00278
    if re.fullmatch(r"\d+", s):
        return "S" + s.zfill(5)
    m = re.fullmatch(r"[Ss](\d+)", s)
    if m:
        return "S" + m.group(1).zfill(5)
    return s


def clean_id(x) -> str:
    """Turn 501.0 -> '501', keep '19A' as '19A', None -> None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(".0", "")
    return s


def add_time_features(df):
    dt = pd.to_datetime(df["feed_timestamp"], unit="s", utc=True)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["stopped_flag"] = (df["speed_mps"] < 0.5).astype(int)
    return df


def pick_latest_tripupdates_snapshot(index_csv: str):
    idx = pd.read_csv(index_csv)
    if idx.empty:
        raise RuntimeError(f"{index_csv} is empty. Capture tripupdates first.")

    # Use the last row (latest capture)
    row = idx.sort_values("capture_ts").iloc[-1]
    pb_path = str(row["pb_path"])
    veh_ts = int(row["vehicle_feed_ts"])
    tu_ts = int(row["tripupdates_feed_ts"])
    cap_ts = int(row["capture_ts"])
    return pb_path, veh_ts, tu_ts, cap_ts


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# ---------- main ----------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run model1.py first (and ensure it saves eta_model.joblib)."
        )

    if not os.path.exists(TRIPUPDATES_INDEX):
        raise FileNotFoundError(
            f"Missing {TRIPUPDATES_INDEX}. Your capture script should write it."
        )

    # 0) pick the latest tripupdates snapshot that matches vehicle feed timestamp
    tu_pb_path, veh_feed_ts, tu_feed_ts, cap_ts = pick_latest_tripupdates_snapshot(TRIPUPDATES_INDEX)
    if not os.path.exists(tu_pb_path):
        raise FileNotFoundError(f"TripUpdates pb not found: {tu_pb_path}")

    # 1) vehicle snapshot aligned to that vehicle feed timestamp
    vdf = pd.read_csv(VEH_LOG)
    if "feed_timestamp" not in vdf.columns:
        raise RuntimeError("vehicle_positions_log.csv missing feed_timestamp column")

    # pick that exact feed timestamp if it exists; otherwise fallback to nearest
    if (vdf["feed_timestamp"] == veh_feed_ts).any():
        snap = vdf[vdf["feed_timestamp"] == veh_feed_ts].copy()
        used_ts = veh_feed_ts
    else:
        # nearest timestamp fallback (rare)
        vdf["_abs_diff"] = (vdf["feed_timestamp"] - veh_feed_ts).abs()
        nearest = vdf.loc[vdf["_abs_diff"].idxmin(), "feed_timestamp"]
        snap = vdf[vdf["feed_timestamp"] == nearest].copy()
        used_ts = int(nearest)

    # normalize IDs + required fields
    snap["trip_id"] = snap.get("trip_id", None).apply(clean_id)
    snap["route_id"] = snap.get("route_id", None).apply(clean_id)
    snap["stop_id"] = snap.get("stop_id", None).apply(norm_stop_id)

    # some logs store current_stop_sequence as float; fix it
    snap["current_stop_sequence"] = pd.to_numeric(snap.get("current_stop_sequence", np.nan), errors="coerce")
    snap = snap.dropna(subset=["trip_id", "route_id", "current_stop_sequence", "dist_to_stop_m", "speed_mps"]).copy()
    snap["current_stop_sequence"] = snap["current_stop_sequence"].astype(int)

    # ensure trip_start_date exists; fallback to local date of snapshot
    if "trip_start_date" not in snap.columns or snap["trip_start_date"].isna().all():
        local_date = datetime.fromtimestamp(int(used_ts), tz=UTC).astimezone(TZ).strftime("%Y%m%d")
        snap["trip_start_date"] = local_date
    snap["trip_start_date"] = snap["trip_start_date"].apply(clean_id)

    # 2) scheduled lookup from GTFS static (trip_id + stop_sequence -> sched_arrival_sec)
    st = load_stop_times_lookup(GTFS_ZIP).copy()
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_sequence"] = st["stop_sequence"].astype(int)

    snap["trip_id"] = snap["trip_id"].astype(str)
    snap = snap.merge(
        st,
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left",
        suffixes=("", "_gtfs"),
    )
    snap = snap.dropna(subset=["sched_arrival_sec"]).copy()

    # scheduled arrival unix
    snap["sched_arrival_ts"] = snap.apply(
        lambda r: scheduled_unix_from_start_date(r["trip_start_date"], int(r["sched_arrival_sec"])),
        axis=1,
    )

    # 3) TripUpdates (official RT prediction)
    tu = parse_tripupdates_pb(tu_pb_path).copy()
    tu["trip_id"] = tu["trip_id"].astype(str)
    tu["stop_sequence"] = pd.to_numeric(tu["stop_sequence"], errors="coerce").astype("Int64")
    tu["stop_id"] = tu.get("stop_id", None).apply(norm_stop_id)

    tu = tu.dropna(subset=["trip_id", "stop_sequence", "rt_pred_arrival_ts"]).copy()
    tu["stop_sequence"] = tu["stop_sequence"].astype(int)

    # merge on trip_id + stop_sequence (best common key)
    snap = snap.merge(
        tu[["trip_id", "stop_sequence", "rt_pred_arrival_ts"]],
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left",
        suffixes=("", "_tu"),
    )

    # measure match rate
    match_rate = float(snap["rt_pred_arrival_ts"].notna().mean()) if len(snap) else 0.0

    # 4) AI model prediction
    pipe = joblib.load(MODEL_PATH)

    snap = add_time_features(snap)

    # features the model expects; fill if missing
    for col in ["progress_rate", "progress_rate_ewm", "speed_ewm", "bearing", "congestion_level"]:
        if col not in snap.columns:
            snap[col] = 0.0

    # make sure stop_id + route_id exist for model categorical inputs
    snap["route_id"] = snap["route_id"].fillna("UNK").astype(str)
    snap["stop_id"] = snap["stop_id"].fillna("UNK").astype(str)

    num_features = [
        "dist_to_stop_m", "speed_mps", "speed_ewm", "progress_rate", "progress_rate_ewm",
        "bearing", "current_stop_sequence", "congestion_level",
        "hour", "dow", "is_weekend", "stopped_flag",
    ]
    cat_features = ["route_id", "stop_id"]

    for c in num_features:
        snap[c] = pd.to_numeric(snap[c], errors="coerce").fillna(0.0)

    eta_pred_s = pipe.predict(snap[num_features + cat_features])
    eta_pred_s = np.maximum(eta_pred_s, 0)  # no negative ETAs
    snap["ai_pred_arrival_ts"] = snap["feed_timestamp"] + eta_pred_s

    # 5) Compute times + ETAs + delays
    snapshot_local = pd.to_datetime(int(used_ts), unit="s", utc=True).tz_convert(TZ)
    snap["snapshot_local"] = snapshot_local

    snap["sched_time"] = pd.to_datetime(snap["sched_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)
    snap["ai_time"] = pd.to_datetime(snap["ai_pred_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)

    snap["sched_eta_min"] = (snap["sched_arrival_ts"] - snap["feed_timestamp"]) / 60.0
    snap["ai_eta_min"] = (snap["ai_pred_arrival_ts"] - snap["feed_timestamp"]) / 60.0
    snap["ai_delay"] = (snap["ai_pred_arrival_ts"] - snap["sched_arrival_ts"]) / 60.0

    snap["rt_time"] = pd.to_datetime(snap["rt_pred_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)
    snap["rt_eta_min"] = (snap["rt_pred_arrival_ts"] - snap["feed_timestamp"]) / 60.0
    snap["rt_delay"] = (snap["rt_pred_arrival_ts"] - snap["sched_arrival_ts"]) / 60.0

    # 6) Pick a nice output subset (soonest scheduled arrivals)
    out = snap.copy()

    # Keep stop_name if present
    if "stop_name" not in out.columns:
        out["stop_name"] = ""

    out = out[[
        "route_id", "stop_id", "stop_name",
        "snapshot_local",
        "sched_time", "sched_eta_min",
        "rt_time", "rt_eta_min", "rt_delay",
        "ai_time", "ai_eta_min", "ai_delay",
        "dist_to_stop_m", "speed_mps",
    ]].copy()

    # rename for the nicer headers you used
    out = out.rename(columns={
        "dist_to_stop_m": "dist_m",
        "speed_mps": "speed",
    })

    # sort: show things that should be arriving soon (by scheduled time)
    out = out.sort_values("sched_time").head(25)

    # 7) Save CSV (raw-ish)
    out.to_csv(OUT_CSV, index=False)

    # 8) Save HTML (readable)
    def fmt_dt(x):
        if pd.isna(x):
            return ""
        return str(x)

    html_df = out.copy()
    for c in ["snapshot_local", "sched_time", "rt_time", "ai_time"]:
        html_df[c] = html_df[c].apply(fmt_dt)

    # round numeric columns for display
    for c in ["sched_eta_min", "rt_eta_min", "rt_delay", "ai_eta_min", "ai_delay", "dist_m", "speed"]:
        if c in html_df.columns:
            html_df[c] = pd.to_numeric(html_df[c], errors="coerce").round(2)

    title = "Delay Comparison (Schedule vs Official RT vs Our AI)"
    meta = f"""
    <h1>{title}</h1>
    <p><b>Snapshot (local):</b> {snapshot_local}</p>
    <p><b>TripUpdates snapshot:</b> {tu_pb_path}</p>
    <p><b>TripUpdates match rate:</b> {match_rate*100:.1f}%</p>
    <hr/>
    """

    table_html = html_df.to_html(index=False, escape=False)
    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 18px; }
      table { border-collapse: collapse; width: 100%; font-size: 14px; }
      th, td { border: 1px solid #e6e6e6; padding: 8px 10px; vertical-align: top; }
      th { background: #f6f6f6; text-align: left; position: sticky; top: 0; }
      tr:nth-child(even) { background: #fbfbfb; }
      .note { color: #666; font-size: 13px; }
    </style>
    """

    note = """
    <p class="note">
      Notes:
      <br/>• sched_eta_min / rt_eta_min / ai_eta_min are “minutes from snapshot until arrival” (negative means it was scheduled/predicted in the past).
      <br/>• rt_delay / ai_delay are “minutes late vs schedule” (negative means early).
    </p>
    """

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("<html><head>")
        f.write(style)
        f.write("</head><body>")
        f.write(meta)
        f.write(table_html)
        f.write(note)
        f.write("</body></html>")

    print("\n=== Delay Comparison (Schedule vs Official RT vs Our AI) ===")
    print(f"Snapshot time (local): {snapshot_local}")
    print(f"TripUpdates snapshot: {tu_pb_path}")
    print(f"TripUpdates match rate (shown rows): {match_rate*100:.1f}%")
    print(f"\nWrote:\n  - {OUT_CSV}\n  - {OUT_HTML}\n")


if __name__ == "__main__":
    main()
