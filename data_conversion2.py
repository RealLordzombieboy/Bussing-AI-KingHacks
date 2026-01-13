from google.transit import gtfs_realtime_pb2
import requests
import csv
import os
import time

FEED_URL = "https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb"
OUT_CSV = "vehicle_positions_log.csv"
INTERVAL_SEC = 30  

def fetch_feed(timeout=20) -> gtfs_realtime_pb2.FeedMessage:
    r = requests.get(FEED_URL, timeout=timeout)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed

def feed_to_rows(feed: gtfs_realtime_pb2.FeedMessage):
    rows = []
    for e in feed.entity:
        if not e.HasField("vehicle"):
            continue

        v = e.vehicle
        has_pos = v.HasField("position")
        has_trip = v.HasField("trip")

        rows.append({
            "feed_timestamp": int(feed.header.timestamp),  # snapshot timestamp
            "entity_id": e.id,

            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
            "route_id": v.trip.route_id if (has_trip and v.trip.HasField("route_id")) else None,
            "trip_id": v.trip.trip_id if (has_trip and v.trip.HasField("trip_id")) else None,

            "stop_id": v.stop_id if v.HasField("stop_id") else None,
            "current_stop_sequence": v.current_stop_sequence if v.HasField("current_stop_sequence") else None,
            "current_status": int(v.current_status) if v.HasField("current_status") else None,
            "congestion_level": int(v.congestion_level) if v.HasField("congestion_level") else None,

            "vehicle_timestamp": int(v.timestamp) if v.HasField("timestamp") else None,

            "lat": v.position.latitude if has_pos else None,
            "lon": v.position.longitude if has_pos else None,
            "speed_mps": v.position.speed if (has_pos and v.position.HasField("speed")) else None,
            "bearing": v.position.bearing if (has_pos and v.position.HasField("bearing")) else None,
            "odometer": v.position.odometer if (has_pos and v.position.HasField("odometer")) else None,
        })
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
    last_ts = None
    print(f"Logging to {OUT_CSV} every {INTERVAL_SEC}s. Ctrl+C to stop.")

    while True:
        try:
            feed = fetch_feed()
            ts = int(feed.header.timestamp)

            if last_ts is not None and ts == last_ts:
                print("No new feed update (same timestamp).")
            else:
                rows = feed_to_rows(feed)
                n = append_rows(rows)
                print(f"Appended {n} rows @ feed_timestamp={ts}")
                last_ts = ts

        except Exception as e:
            print("Error:", repr(e))

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
