from google.transit import gtfs_realtime_pb2
from collections import Counter
from datetime import datetime, timezone

pb_path = "vehicleupdates.pb"

feed = gtfs_realtime_pb2.FeedMessage()
with open(pb_path, "rb") as f:
    feed.ParseFromString(f.read())

print("Feed timestamp (UTC):", datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc))
print("Entities:", len(feed.entity))

c = Counter()
n = 0

for e in feed.entity:
    if not e.HasField("vehicle"):
        continue
    n += 1
    v = e.vehicle

    if v.HasField("position"): c["position"] += 1
    if v.position.HasField("speed"): c["speed"] += 1
    if v.position.HasField("odometer"): c["odometer"] += 1
    if v.HasField("timestamp"): c["timestamp"] += 1
    if v.HasField("current_status"): c["current_status"] += 1
    if v.HasField("stop_id"): c["stop_id"] += 1
    if v.HasField("trip"): c["trip"] += 1
    if v.HasField("vehicle"): c["vehicle_id"] += 1

    # If trip exists, check what's inside
    if v.HasField("trip"):
        if v.trip.HasField("trip_id"): c["trip_id"] += 1
        if v.trip.HasField("route_id"): c["route_id"] += 1
        if v.trip.HasField("direction_id"): c["direction_id"] += 1

print("\nField coverage:")
for k in sorted(c.keys()):
    print(f"{k:14s} {c[k]:3d}/{n}  ({c[k]/n:.0%})")
