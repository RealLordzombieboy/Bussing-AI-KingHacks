# live_api.py
import time
import numpy as np
import pandas as pd

from delay_comparison import build_delay_table, write_outputs, write_live_json

OUT_HTML = "delay_comparison_latest.html"
OUT_CSV = "delay_comparison_latest.csv"
OUT_JSON = "live_map.json"

INTERVAL_SEC = 10  # how often the map/table refreshes


# ---------- formatting helpers ----------
def _to_ts(x):
    """Robust: accept pandas Timestamp, python datetime, or string; return pandas Timestamp or NaT."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT


def fmt_hhmm(x):
    t = _to_ts(x)
    if pd.isna(t):
        return None
    # if timezone-aware, strftime works; if naive, still works
    return t.strftime("%H:%M")


def fmt_delay_min(dmin):
    if dmin is None or (isinstance(dmin, float) and np.isnan(dmin)):
        return None
    try:
        dmin = float(dmin)
    except Exception:
        return None
    sign = "+" if dmin >= 0 else ""
    return f"{sign}{dmin:.1f}m"


def fmt_label(arrival_time, delay_min, prefix=None):
    """Return like '18:26 (+1.4m)' or None if missing."""
    hhmm = fmt_hhmm(arrival_time)
    d = fmt_delay_min(delay_min)
    if hhmm is None or d is None:
        return None
    s = f"{hhmm} ({d})"
    if prefix:
        return f"{prefix}: {s}"
    return s


def main():
    print(f"Live demo generator running every {INTERVAL_SEC}s. Ctrl+C to stop.")
    while True:
        try:
            out, meta, _ = build_delay_table()

            # keep a manageable number
            out = out.sort_values("sched_time").head(60).copy()

            # ---- Add compact labels for the map UI ----
            # expected columns from delay_comparison.py output:
            # sched_time, rt_time, ai_time, rt_delay, ai_delay
            if "sched_time" in out.columns:
                out["sched_hhmm"] = out["sched_time"].apply(fmt_hhmm)
            else:
                out["sched_hhmm"] = None

            out["rt_label"] = out.apply(lambda r: fmt_label(r.get("rt_time"), r.get("rt_delay")), axis=1)
            out["ai_label"] = out.apply(lambda r: fmt_label(r.get("ai_time"), r.get("ai_delay")), axis=1)

            # optional: also provide these (nice for UI)
            out["rt_hhmm"] = out["rt_time"].apply(fmt_hhmm) if "rt_time" in out.columns else None
            out["ai_hhmm"] = out["ai_time"].apply(fmt_hhmm) if "ai_time" in out.columns else None

            # write the human-readable table outputs
            write_outputs(out, meta, out_csv=OUT_CSV, out_html=OUT_HTML)

            # write live JSON for the map (we assume write_live_json uses the columns in `out`)
            write_live_json(out, meta, json_path=OUT_JSON)

            print(
                f"[OK] Updated at snapshot_local={meta.get('snapshot_local')}  "
                f"match_rate={meta.get('match_rate',0)*100:.1f}%"
            )

        except Exception as e:
            print("[ERR]", repr(e))

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
