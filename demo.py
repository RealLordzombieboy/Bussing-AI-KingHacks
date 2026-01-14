import pandas as pd
import numpy as np
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

CURRENT = "current_data.csv"
STOPS = "bus_stops.csv"
TZ = ZoneInfo("America/Toronto")

def norm_stop_id(s: str) -> str:
    s = str(s).strip()
    if re.fullmatch(r"\d+", s):
        return "S" + s.zfill(5)
    m = re.fullmatch(r"[Ss](\d+)", s)
    if m:
        return "S" + m.group(1).zfill(5)
    return s

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def main():
    df = pd.read_csv(CURRENT, header=None)

    if df.iloc[0].astype(str).str.contains("entity_id|vehicle_id|trip_id|route_id", regex=True).any():
        df = df.iloc[1:].copy()

    df = df.iloc[:, :6]
    df.columns = ["trip_id", "route_id", "stop_id_raw", "speed_mps", "start_time", "vehicle_id"]

    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
    df["stop_id"] = df["stop_id_raw"].apply(norm_stop_id)
    df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")

    # 2) Load stops lookup (stop_id -> lat/lon/name)
    stops = pd.read_csv(STOPS)
    stops["stop_id"] = stops["Stop ID"].apply(norm_stop_id)
    stops = stops.rename(columns={"STOP_LAT":"stop_lat", "STOP_LONG":"stop_lon", "Stop Name":"stop_name"})
    stops = stops.dropna(subset=["stop_id","stop_lat","stop_lon"])
    stops = stops.drop_duplicates(subset=["stop_id"], keep="first")
    stop_lookup = stops.set_index("stop_id")[["stop_lat","stop_lon","stop_name"]]

    # 3) Join stop info (many rows may not match; that's fine)
    df = df.join(stop_lookup, on="stop_id")

    remaining_m = np.where(df["stop_name"].notna(), 600.0, 1200.0)  # heuristic
    df["eta_seconds"] = remaining_m / np.maximum(df["speed_mps"].fillna(0), 1.0)

    # 5) Pretty-print top 10 by soonest ETA
    now = datetime.now(tz=TZ)
    df["eta_minutes"] = df["eta_seconds"] / 60.0
    df = df.sort_values("eta_seconds").head(10)

    print("\nQuick demo (proxy ETA) — shows output format your app will use")
    print("-"*90)
    for _, r in df.iterrows():
        arrival = now + pd.to_timedelta(r["eta_seconds"], unit="s")
        stop_label = r["stop_name"] if pd.notna(r["stop_name"]) else f"stop {r['stop_id']}"
        print(
            f"Route {r['route_id']:>4} | Bus {r['vehicle_id']} -> {stop_label} "
            f"| ~{r['eta_minutes']:.1f} min | ~{arrival.strftime('%-I:%M %p')}"
        )

if __name__ == "__main__":
    main()
