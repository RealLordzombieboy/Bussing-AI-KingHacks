from google.transit import gtfs_realtime_pb2
import requests
import csv
import os
import time
import pandas as pd
import math
import re

FEED_URL = "https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb"
OUT_CSV = "vehicle_positions_log.csv"
INTERVAL_SEC = 30

STOPS_CSV = "bus_stops.csv"


def norm_stop_id(s: str) -> str:
    """Normalize stop IDs so formats like '00254' and 'S254' become 'S00254'."""
    s = str(s).strip()
    if re.fullmatch(r"\d+", s):
        return "S" + s.zfill(5)
    m = re.fullmatch(r"[Ss](\d+)", s)
    if m:
        return "S" + m.group(1).zfill(5)
    return s


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_stop_lookup():
    stops = pd.read_csv(STOPS_CSV)

    stops["stop_id_norm"] = stops["Stop ID"].apply(norm_stop_id)

    stops = stops.rename(
        columns={
            "STOP_LAT": "stop_lat",
            "STOP_LONG": "stop_lon",
            "Stop Name": "stop_name",
        }
    )

    stops = stops.dropna(subset=["stop_id_norm", "stop_lat", "stop_lon"])
    stops = stops.drop_duplicates(subset=["stop_id_norm"], keep="first")

    return (
        stops.set_index("stop_id_norm")[["stop_lat", "stop_lon", "stop_name"]]
        .to_dict("index")
    )


def fetch_feed(timeout=20) -> gtfs_realtime_pb2.FeedMessage:
    r = requests.get(FEED_URL, timeout=timeout)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed


def feed_to_rows(feed: gtfs_realtime_pb2.FeedMessage, stop_lookup: dict):
    rows = []
    for e in feed.entity:
        if not e.HasField("vehicle"):
            continue

        v = e.vehicle
        has_pos = v.HasField("position")
        has_trip = v.HasField("trip")

        # Trip meta (IMPORTANT for matching TripUpdates)
        trip_start_date = v.trip.start_date if (has_trip and v.trip.HasField("start_date")) else None
        trip_start_time = v.trip.start_time if (has_trip and v.trip.HasField("start_time")) else None

        # Normalize stop_id if present
        stop_id_raw = v.stop_id if v.HasField("stop_id") else None
        stop_id = norm_stop_id(stop_id_raw) if stop_id_raw is not None else None

        stop_name = None
        dist_to_stop_m = None

        if stop_id and has_pos and stop_id in stop_lookup:
            stop_lat = float(stop_lookup[stop_id]["stop_lat"])
            stop_lon = float(stop_lookup[stop_id]["stop_lon"])
            stop_name = stop_lookup[stop_id]["stop_name"]
            dist_to_stop_m = haversine_m(
                v.position.latitude,
                v.position.longitude,
                stop_lat,
                stop_lon,
            )

        rows.append(
            {
                "feed_timestamp": int(feed.header.timestamp),
                "entity_id": e.id,

                "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,

                "route_id": v.trip.route_id if (has_trip and v.trip.HasField("route_id")) else None,
                "trip_id": v.trip.trip_id if (has_trip and v.trip.HasField("trip_id")) else None,

                # NEW:
                "trip_start_date": trip_start_date,
                "trip_start_time": trip_start_time,

                "stop_id": stop_id,
                "stop_name": stop_name,
                "dist_to_stop_m": dist_to_stop_m,

                "current_stop_sequence": v.current_stop_sequence if v.HasField("current_stop_sequence") else None,
                "current_status": int(v.current_status) if v.HasField("current_status") else None,
                "congestion_level": int(v.congestion_level) if v.HasField("congestion_level") else None,

                "vehicle_timestamp": int(v.timestamp) if v.HasField("timestamp") else None,

                "lat": v.position.latitude if has_pos else None,
                "lon": v.position.longitude if has_pos else None,
                "speed_mps": v.position.speed if (has_pos and v.position.HasField("speed")) else None,
                "bearing": v.position.bearing if (has_pos and v.position.HasField("bearing")) else None,
                "odometer": v.position.odometer if (has_pos and v.position.HasField("odometer")) else None,
            }
        )
    return rows


def append_rows(rows, out_path=OUT_CSV):
    if not rows:
        return 0

    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    stop_lookup = load_stop_lookup()
    print("Stops loaded:", len(stop_lookup))
    print(f"Logging to {OUT_CSV} every {INTERVAL_SEC}s. Ctrl+C to stop.")

    last_ts = None

    while True:
        try:
            feed = fetch_feed()
            ts = int(feed.header.timestamp)

            if last_ts is not None and ts == last_ts:
                print("No new feed update (same timestamp).")
            else:
                rows = feed_to_rows(feed, stop_lookup)
                n = append_rows(rows)
                print(f"Appended {n} rows @ feed_timestamp={ts}")
                last_ts = ts

        except Exception as e:
            print("Error:", repr(e))

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
