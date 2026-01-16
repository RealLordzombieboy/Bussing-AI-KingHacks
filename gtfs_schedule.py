import zipfile
import pandas as pd

def gtfs_time_to_seconds(t: str) -> int:
    # supports "25:10:00"
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def load_stop_times_lookup(gtfs_zip_path: str):
    """
    Returns a DataFrame with:
      trip_id, stop_sequence, stop_id, sched_arrival_sec
    """
    with zipfile.ZipFile(gtfs_zip_path) as z:
        stop_times = pd.read_csv(z.open("stop_times.txt"), dtype={"trip_id": str, "stop_id": str})

    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.dropna(subset=["stop_sequence", "arrival_time"])
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)

    stop_times["sched_arrival_sec"] = stop_times["arrival_time"].astype(str).apply(gtfs_time_to_seconds)

    return stop_times[["trip_id", "stop_sequence", "stop_id", "sched_arrival_sec"]]
