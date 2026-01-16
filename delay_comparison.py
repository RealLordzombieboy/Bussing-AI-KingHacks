# delay_comparison.py
import os
import re
import json
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
OUT_JSON = "live_map.json"  # for the map

TZ = ZoneInfo("America/Toronto")
UTC = ZoneInfo("UTC")


# ---------- helpers ----------
def yyyymmdd_to_date(s: str):
    s = str(s).strip()
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

    row = idx.sort_values("capture_ts").iloc[-1]
    pb_path = str(row["pb_path"])
    veh_ts = int(row["vehicle_feed_ts"])
    tu_ts = int(row["tripupdates_feed_ts"])
    cap_ts = int(row["capture_ts"])
    return pb_path, veh_ts, tu_ts, cap_ts


def _fmt_hhmm(dt_val) -> str | None:
    """Return 'HH:MM' in local timezone for display."""
    if pd.isna(dt_val):
        return None
    try:
        # dt_val is already tz-aware in our pipeline
        return pd.Timestamp(dt_val).strftime("%H:%M")
    except Exception:
        return None


def _fmt_label(time_dt, delay_min) -> str | None:
    """
    Example: "18:26 (+1.4m)" or "18:26 (-0.8m)"
    If time missing -> None.
    If delay missing -> "18:26" (still show time).
    """
    t = _fmt_hhmm(time_dt)
    if not t:
        return None
    if pd.isna(delay_min):
        return t
    try:
        d = float(delay_min)
        sign = "+" if d >= 0 else ""
        return f"{t} ({sign}{d:.1f}m)"
    except Exception:
        return t


def build_delay_table(
    gtfs_zip=GTFS_ZIP,
    veh_log=VEH_LOG,
    tripupdates_index=TRIPUPDATES_INDEX,
    model_path=MODEL_PATH,
):
    """
    Returns:
      - out_df: human table (for CSV/HTML/map JSON)
      - meta: snapshot info + match rate
      - snap_full: full merged snapshot
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run model1.py first (and save eta_model.joblib).")
    if not os.path.exists(tripupdates_index):
        raise FileNotFoundError(f"Missing {tripupdates_index}. Your capture script should write it.")
    if not os.path.exists(veh_log):
        raise FileNotFoundError(f"Missing {veh_log}. Run data_conversion2.py / capture first.")

    tu_pb_path, veh_feed_ts, tu_feed_ts, cap_ts = pick_latest_tripupdates_snapshot(tripupdates_index)
    if not os.path.exists(tu_pb_path):
        raise FileNotFoundError(f"TripUpdates pb not found: {tu_pb_path}")

    # 1) vehicle snapshot aligned to vehicle_feed_ts
    vdf = pd.read_csv(veh_log)
    if "feed_timestamp" not in vdf.columns:
        raise RuntimeError("vehicle_positions_log.csv missing feed_timestamp column")

    if (vdf["feed_timestamp"] == veh_feed_ts).any():
        snap = vdf[vdf["feed_timestamp"] == veh_feed_ts].copy()
        used_ts = veh_feed_ts
    else:
        vdf["_abs_diff"] = (vdf["feed_timestamp"] - veh_feed_ts).abs()
        nearest = vdf.loc[vdf["_abs_diff"].idxmin(), "feed_timestamp"]
        snap = vdf[vdf["feed_timestamp"] == nearest].copy()
        used_ts = int(nearest)

    # normalize IDs
    snap["trip_id"] = snap.get("trip_id", None).apply(clean_id)
    snap["route_id"] = snap.get("route_id", None).apply(clean_id)
    snap["stop_id"] = snap.get("stop_id", None).apply(norm_stop_id)

    snap["current_stop_sequence"] = pd.to_numeric(snap.get("current_stop_sequence", np.nan), errors="coerce")
    snap = snap.dropna(subset=["trip_id", "route_id", "current_stop_sequence", "dist_to_stop_m", "speed_mps"]).copy()
    snap["current_stop_sequence"] = snap["current_stop_sequence"].astype(int)

    # ensure trip_start_date exists
    if "trip_start_date" not in snap.columns or snap["trip_start_date"].isna().all():
        local_date = datetime.fromtimestamp(int(used_ts), tz=UTC).astimezone(TZ).strftime("%Y%m%d")
        snap["trip_start_date"] = local_date
    snap["trip_start_date"] = snap["trip_start_date"].apply(clean_id)

    # 2) scheduled lookup from GTFS static
    st = load_stop_times_lookup(gtfs_zip).copy()
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

    snap["sched_arrival_ts"] = snap.apply(
        lambda r: scheduled_unix_from_start_date(r["trip_start_date"], int(r["sched_arrival_sec"])),
        axis=1,
    )

    # 3) TripUpdates (official RT)
    tu = parse_tripupdates_pb(tu_pb_path).copy()
    tu["trip_id"] = tu["trip_id"].astype(str)
    tu["stop_sequence"] = pd.to_numeric(tu["stop_sequence"], errors="coerce").astype("Int64")
    tu["stop_id"] = tu.get("stop_id", None).apply(norm_stop_id)

    tu = tu.dropna(subset=["trip_id", "stop_sequence", "rt_pred_arrival_ts"]).copy()
    tu["stop_sequence"] = tu["stop_sequence"].astype(int)

    snap = snap.merge(
        tu[["trip_id", "stop_sequence", "rt_pred_arrival_ts"]],
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="left",
        suffixes=("", "_tu"),
    )

    match_rate = float(snap["rt_pred_arrival_ts"].notna().mean()) if len(snap) else 0.0

    # 4) AI model prediction
    pipe = joblib.load(model_path)
    snap = add_time_features(snap)

    for col in ["progress_rate", "progress_rate_ewm", "speed_ewm", "bearing", "congestion_level"]:
        if col not in snap.columns:
            snap[col] = 0.0

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
    eta_pred_s = np.maximum(eta_pred_s, 0)
    snap["ai_pred_arrival_ts"] = snap["feed_timestamp"] + eta_pred_s

    # 5) times, etas, delays
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

    # output subset
    out = snap.copy()
    if "stop_name" not in out.columns:
        out["stop_name"] = ""

    out = out[[
        "route_id", "stop_id", "stop_name",
        "snapshot_local",
        "sched_time", "sched_eta_min",
        "rt_time", "rt_eta_min", "rt_delay",
        "ai_time", "ai_eta_min", "ai_delay",
        "dist_to_stop_m", "speed_mps",
        "vehicle_id", "lat", "lon", "trip_id",
    ]].copy()

    out = out.rename(columns={
        "dist_to_stop_m": "dist_m",
        "speed_mps": "speed",
    })

    # --- NEW: compact display strings for the map popup ---
    out["sched_hhmm"] = out["sched_time"].apply(_fmt_hhmm)
    out["rt_label"] = out.apply(lambda r: _fmt_label(r["rt_time"], r["rt_delay"]), axis=1)
    out["ai_label"] = out.apply(lambda r: _fmt_label(r["ai_time"], r["ai_delay"]), axis=1)

    meta = {
        "snapshot_local": str(snapshot_local),
        "vehicle_feed_ts_used": int(used_ts),
        "veh_feed_ts_target": int(veh_feed_ts),
        "tripupdates_feed_ts": int(tu_feed_ts),
        "capture_ts": int(cap_ts),
        "tripupdates_pb_path": tu_pb_path,
        "match_rate": match_rate,
    }

    return out, meta, snap


def write_outputs(out: pd.DataFrame, meta: dict, out_csv=OUT_CSV, out_html=OUT_HTML):
    out.to_csv(out_csv, index=False)

    def fmt_dt(x):
        if pd.isna(x):
            return ""
        return str(x)

    html_df = out.copy()
    for c in ["snapshot_local", "sched_time", "rt_time", "ai_time"]:
        if c in html_df.columns:
            html_df[c] = html_df[c].apply(fmt_dt)

    for c in ["sched_eta_min", "rt_eta_min", "rt_delay", "ai_eta_min", "ai_delay", "dist_m", "speed"]:
        if c in html_df.columns:
            html_df[c] = pd.to_numeric(html_df[c], errors="coerce").round(2)

    title = "Delay Comparison (Schedule vs Official RT vs Our AI)"
    meta_html = f"""
    <h1>{title}</h1>
    <p><b>Snapshot (local):</b> {meta.get("snapshot_local")}</p>
    <p><b>TripUpdates snapshot:</b> {meta.get("tripupdates_pb_path")}</p>
    <p><b>TripUpdates match rate:</b> {meta.get("match_rate", 0)*100:.1f}%</p>
    <hr/>
    """

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
      <br/>• sched_eta_min / rt_eta_min / ai_eta_min are “minutes from snapshot until arrival” (negative means the scheduled/predicted time is already in the past).
      <br/>• rt_delay / ai_delay are “minutes late vs schedule” (negative means early).
    </p>
    """

    table_html = html_df.to_html(index=False, escape=False)

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<html><head>")
        f.write(style)
        f.write("</head><body>")
        f.write(meta_html)
        f.write(table_html)
        f.write(note)
        f.write("</body></html>")


def write_live_json(out: pd.DataFrame, meta: dict, json_path=OUT_JSON):
    """
    Writes a compact JSON file for the map.
    IMPORTANT: include sched_hhmm / rt_label / ai_label so the popup isn't cluttered.
    """
    df = out.dropna(subset=["lat", "lon"]).copy()

    def _num(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _str(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        s = str(x)
        if s == "NaT":
            return None
        return s

    markers = []
    for _, r in df.iterrows():
        markers.append({
            "lat": _num(r.get("lat")),
            "lon": _num(r.get("lon")),

            "route_id": _str(r.get("route_id")),
            "vehicle_id": _str(r.get("vehicle_id")),
            "trip_id": _str(r.get("trip_id")),

            "stop_id": _str(r.get("stop_id")),
            "stop_name": _str(r.get("stop_name")),

            "dist_m": _num(r.get("dist_m")),
            "speed": _num(r.get("speed")),

            # compact popup fields
            "sched_hhmm": _str(r.get("sched_hhmm")),
            "rt_label": _str(r.get("rt_label")),
            "ai_label": _str(r.get("ai_label")),

            # keep numeric values for coloring/debugging
            "rt_delay": _num(r.get("rt_delay")),
            "ai_delay": _num(r.get("ai_delay")),
            "ai_eta_min": _num(r.get("ai_eta_min")),
        })

    payload = {"meta": meta, "markers": markers}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    out, meta, _snap_full = build_delay_table()
    out = out.sort_values("sched_time").head(60)

    write_outputs(out, meta, out_csv=OUT_CSV, out_html=OUT_HTML)
    write_live_json(out, meta, json_path=OUT_JSON)

    print("\n=== Delay Comparison (Schedule vs Official RT vs Our AI) ===")
    print(f"Snapshot time (local): {meta.get('snapshot_local')}")
    print(f"TripUpdates snapshot: {meta.get('tripupdates_pb_path')}")
    print(f"TripUpdates match rate (shown rows): {meta.get('match_rate', 0)*100:.1f}%")
    print(f"\nWrote:\n  - {OUT_CSV}\n  - {OUT_HTML}\n  - {OUT_JSON}\n")


if __name__ == "__main__":
    main()
