from google.transit import gtfs_realtime_pb2
import math
import pandas as pd

stops = pd.read_csv("bus_stops.csv")  
print(stops.columns)

# Make a lookup: stop_id -> (lat, lon, name)
stop_lookup = (
    stops
    .rename(columns={
        "Stop ID": "stop_id",
        "STOP_LAT": "stop_lat",
        "STOP_LONG": "stop_lon",
        "Stop Name": "stop_name"
    })
    [["stop_id", "stop_lat", "stop_lon", "stop_name"]]
    .dropna(subset=["stop_id", "stop_lat", "stop_lon"])
)

stop_lookup["stop_id"] = stop_lookup["stop_id"].astype(str)
stop_lookup = stop_lookup.set_index("stop_id").to_dict("index")

print("Stops loaded:", len(stop_lookup))

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

feed = gtfs_realtime_pb2.FeedMessage()
with open("vehicleupdates.pb", "rb") as f:
    feed.ParseFromString(f.read())

rows = []
missing_stop = 0

for e in feed.entity:
    if not e.HasField("vehicle"):
        continue
    v = e.vehicle

    if not (v.HasField("position") and v.HasField("stop_id")):
        continue

    sid = str(v.stop_id)
    if sid not in stop_lookup:
        missing_stop += 1
        continue

    stop_lat = stop_lookup[sid]["stop_lat"]
    stop_lon = stop_lookup[sid]["stop_lon"]

    dist_m = haversine_m(
        v.position.latitude, v.position.longitude,
        float(stop_lat), float(stop_lon)
    )

    rows.append({
        "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
        "route_id": v.trip.route_id if v.HasField("trip") else None,
        "trip_id": v.trip.trip_id if v.HasField("trip") else None,
        "stop_id": sid,
        "stop_name": stop_lookup[sid]["stop_name"],
        "ts": v.timestamp,
        "lat": v.position.latitude,
        "lon": v.position.longitude,
        "speed_mps": v.position.speed if v.position.HasField("speed") else None,
        "bearing": v.position.bearing if v.position.HasField("bearing") else None,
        "current_stop_sequence": v.current_stop_sequence if v.HasField("current_stop_sequence") else None,
        "congestion_level": int(v.congestion_level) if v.HasField("congestion_level") else None,
        "dist_to_stop_m": dist_m,
    })

df = pd.DataFrame(rows)
print(df.head())
print("rows:", len(df), "missing stop_ids:", missing_stop)
