# delay_comparison.py
# Compares:
# 1) Scheduled arrival (GTFS static stop_times)
# 2) Official realtime predicted arrival (GTFS-RT tripupdates)
# 3) Our AI predicted arrival (model)
#
# Outputs a readable table to HTML + CSV (+ optional XLSX), and prints a small preview.

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

from gtfs_schedule import load_stop_times_lookup
from tripupdates_parse import parse_tripupdates_pb

GTFS_ZIP = "gtfs.zip"
VEH_LOG = "vehicle_positions_log.csv"
TRIPUPDATES_PB = "tripupdates.pb"
MODEL_PATH = "eta_model.joblib"

TZ = ZoneInfo("America/Toronto")
UTC = ZoneInfo("UTC")


def clean_yyyymmdd(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = s.replace(".0", "")
    s = "".join(ch for ch in s if ch.isdigit())
    return s


def yyyymmdd_to_date(s: str):
    s = clean_yyyymmdd(s)
    return datetime.strptime(s, "%Y%m%d").date()


def scheduled_unix_from_start_date(start_date_yyyymmdd: str, sched_arrival_sec: int) -> int:
    """
    GTFS times can be > 24h. If sched_arrival_sec = 25:10:00, it means next day.
    start_date is local service date (America/Toronto).
    """
    start_date_yyyymmdd = clean_yyyymmdd(start_date_yyyymmdd)
    base = datetime.combine(yyyymmdd_to_date(start_date_yyyymmdd), datetime.min.time()).replace(tzinfo=TZ)

    day_offset = int(sched_arrival_sec) // 86400
    sec_in_day = int(sched_arrival_sec) % 86400

    dt_local = base + timedelta(days=day_offset, seconds=sec_in_day)
    return int(dt_local.astimezone(UTC).timestamp())


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["feed_timestamp"], unit="s", utc=True)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["stopped_flag"] = (df["speed_mps"] < 0.5).astype(int)
    return df


def ensure_stop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    After merges, pandas may rename stop_id -> stop_id_x/stop_id_y.
    This forces a single canonical stop_id column.
    """
    if "stop_id" in df.columns:
        return df

    candidates = [c for c in ["stop_id_x", "stop_id_y", "stop_id_tu", "stop_id_rt"] if c in df.columns]
    if not candidates:
        raise KeyError("stop_id missing after merges, and no stop_id_x/stop_id_y found.")

    df["stop_id"] = df[candidates[0]]
    return df


def main():
    # 1) Latest vehicle snapshot
    vdf = pd.read_csv(VEH_LOG)
    latest_ts = int(vdf["feed_timestamp"].max())
    snap = vdf[vdf["feed_timestamp"] == latest_ts].copy()

    snap = snap.dropna(
        subset=["trip_id", "route_id", "stop_id", "current_stop_sequence", "dist_to_stop_m", "speed_mps", "feed_timestamp"]
    ).copy()

    snap["trip_id"] = snap["trip_id"].astype(str)
    snap["route_id"] = snap["route_id"].astype(str)
    snap["stop_id"] = snap["stop_id"].astype(str)

    snap["current_stop_sequence"] = pd.to_numeric(snap["current_stop_sequence"], errors="coerce").astype("Int64")
    snap = snap.dropna(subset=["current_stop_sequence"]).copy()
    snap["current_stop_sequence"] = snap["current_stop_sequence"].astype(int)

    snapshot_local = datetime.fromtimestamp(latest_ts, tz=UTC).astimezone(TZ)

    # trip_start_date fix
    if "trip_start_date" not in snap.columns or snap["trip_start_date"].isna().all():
        snap["trip_start_date"] = snapshot_local.strftime("%Y%m%d")
    snap["trip_start_date"] = snap["trip_start_date"].apply(clean_yyyymmdd)

    # 2) Scheduled lookup from GTFS static
    st = load_stop_times_lookup(GTFS_ZIP)
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_sequence"] = st["stop_sequence"].astype(int)

    snap = snap.merge(
        st,
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left",
    )

    snap = snap.dropna(subset=["sched_arrival_sec"]).copy()
    snap["sched_arrival_sec"] = snap["sched_arrival_sec"].astype(int)

    snap["sched_arrival_ts"] = snap.apply(
        lambda r: scheduled_unix_from_start_date(r["trip_start_date"], int(r["sched_arrival_sec"])),
        axis=1,
    )

    # 3) TripUpdates
    tu = parse_tripupdates_pb(TRIPUPDATES_PB)
    tu = tu.dropna(subset=["rt_pred_arrival_ts"]).copy()
    tu["trip_id"] = tu["trip_id"].astype(str)
    tu["stop_sequence"] = pd.to_numeric(tu["stop_sequence"], errors="coerce")
    tu = tu.dropna(subset=["stop_sequence"]).copy()
    tu["stop_sequence"] = tu["stop_sequence"].astype(int)

    tu_small = tu[["trip_id", "stop_sequence", "rt_pred_arrival_ts"]].drop_duplicates(["trip_id", "stop_sequence"])

    snap = snap.merge(
        tu_small,
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left",
    )

    snap = ensure_stop_id_column(snap)
    snap["stop_id"] = snap["stop_id"].astype(str)
    snap["route_id"] = snap["route_id"].astype(str)

    # 4) Model prediction
    pipe = joblib.load(MODEL_PATH)
    snap = add_time_features(snap)

    for col in ["progress_rate", "progress_rate_ewm", "speed_ewm"]:
        if col not in snap.columns:
            snap[col] = 0.0

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

    snap[num_features] = snap[num_features].fillna(0)

    eta_pred_s = np.maximum(pipe.predict(snap[num_features + cat_features]), 0)
    snap["ai_eta_s"] = eta_pred_s
    snap["ai_pred_arrival_ts"] = snap["feed_timestamp"] + snap["ai_eta_s"]

    # 5) Compute timestamps + delays
    snap["snapshot_local"] = pd.to_datetime(snap["feed_timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    snap["sched_arrival_local"] = pd.to_datetime(snap["sched_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)
    snap["ai_arrival_local"] = pd.to_datetime(snap["ai_pred_arrival_ts"], unit="s", utc=True).dt.tz_convert(TZ)

    snap["ai_delay_min"] = (snap["ai_pred_arrival_ts"] - snap["sched_arrival_ts"]) / 60.0
    snap["ai_eta_min"] = snap["ai_eta_s"] / 60.0

    snap["rt_delay_min"] = (snap["rt_pred_arrival_ts"] - snap["sched_arrival_ts"]) / 60.0
    snap["rt_arrival_local"] = pd.to_datetime(
        snap["rt_pred_arrival_ts"], unit="s", utc=True, errors="coerce"
    ).dt.tz_convert(TZ)

    # Build output
    out_cols = [
        "route_id",
        "stop_id",
        "stop_name",
        "current_stop_sequence",
        "snapshot_local",
        "sched_arrival_local",
        "rt_arrival_local",
        "ai_arrival_local",
        "dist_to_stop_m",
        "speed_mps",
        "rt_delay_min",
        "ai_delay_min",
        "ai_eta_min",
    ]
    out = snap[out_cols].copy()

    # ---------------------------
    # Better display + save to CSV/HTML
    # ---------------------------
    snap_ts_local = pd.to_datetime(latest_ts, unit="s", utc=True).tz_convert(TZ)

    # Round numeric
    for c in ["dist_to_stop_m", "speed_mps", "rt_delay_min", "ai_delay_min", "ai_eta_min"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    # Strip nanoseconds
    for c in ["snapshot_local", "sched_arrival_local", "rt_arrival_local", "ai_arrival_local"]:
        if c in out.columns:
            out[c] = out[c].dt.floor("s")

    # Add ETA-from-now columns
    out["sched_eta_min"] = ((out["sched_arrival_local"] - snap_ts_local).dt.total_seconds() / 60).round(2)
    out["rt_eta_min"] = ((out["rt_arrival_local"] - snap_ts_local).dt.total_seconds() / 60).round(2)

    # Rename for readability
    out = out.rename(columns={
        "sched_arrival_local": "sched_time",
        "rt_arrival_local": "rt_time",
        "ai_arrival_local": "ai_time",
        "dist_to_stop_m": "dist_m",
        "speed_mps": "speed",
        "rt_delay_min": "rt_delay",
        "ai_delay_min": "ai_delay",
    })

    # Sort by soonest arrival (RT if present, else AI)
    out["sort_eta"] = out["rt_eta_min"].where(out["rt_eta_min"].notna(), out["ai_eta_min"])
    out = out.sort_values("sort_eta").drop(columns=["sort_eta"]).head(25)

    display_cols = [
        "route_id", "stop_id", "stop_name",
        "snapshot_local",
        "sched_time", "sched_eta_min",
        "rt_time", "rt_eta_min", "rt_delay",
        "ai_time", "ai_eta_min", "ai_delay",
        "dist_m", "speed",
    ]
    display_cols = [c for c in display_cols if c in out.columns]
    table = out[display_cols].copy()

    match_rate = table["rt_time"].notna().mean() * 100.0

    # Save files
    OUT_HTML = "delay_comparison_latest.html"
    OUT_CSV = "delay_comparison_latest.csv"

    table.to_csv(OUT_CSV, index=False)

    html = table.to_html(index=False, border=0, justify="left", classes="table")
    html_page = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>Delay Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h2 {{ margin-bottom: 6px; }}
    .meta {{ color: #444; margin-bottom: 14px; }}
    table.table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    table.table th, table.table td {{
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    table.table th {{
      background: #f3f3f3;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h2>Delay Comparison (Schedule vs Official RT vs Our AI)</h2>
  <div class="meta">
    Snapshot time (local): {snapshot_local}<br/>
    TripUpdates match rate (shown rows): {match_rate:.1f}%
  </div>
  {html}
</body>
</html>
"""
    Path(OUT_HTML).write_text(html_page, encoding="utf-8")

    # Terminal preview
    print("\n=== Delay Comparison (Schedule vs Official RT vs Our AI) ===")
    print(f"Snapshot time (local): {snapshot_local}")
    print(f"TripUpdates match rate in shown rows: {match_rate:.1f}%")
    print("\nPreview (top 12 rows):")
    print(table.head(12).to_string(index=False))
    print(f"\nSaved: {OUT_HTML} and {OUT_CSV}")


if __name__ == "__main__":
    main()
