# tripupdates_parse.py
# Parse Kingston GTFS-RT TripUpdates feed (.pb) into a pandas DataFrame.

from google.transit import gtfs_realtime_pb2
import pandas as pd
import re


def norm_stop_id(s: str) -> str:
    """Normalize stop IDs so formats like '00254' and 'S254' become 'S00254'."""
    if s is None:
        return None
    s = str(s).strip()
    if re.fullmatch(r"\d+", s):
        return "S" + s.zfill(5)
    m = re.fullmatch(r"[Ss](\d+)", s)
    if m:
        return "S" + m.group(1).zfill(5)
    return s


def parse_tripupdates_pb(pb_path: str) -> pd.DataFrame:
    feed = gtfs_realtime_pb2.FeedMessage()
    with open(pb_path, "rb") as f:
        feed.ParseFromString(f.read())

    rows = []

    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue

        tu = ent.trip_update
        trip = tu.trip

        trip_id = trip.trip_id if trip.HasField("trip_id") else None
        route_id = trip.route_id if trip.HasField("route_id") else None
        start_date = trip.start_date if trip.HasField("start_date") else None
        start_time = trip.start_time if trip.HasField("start_time") else None

        # Some GTFS-RT feeds include vehicle info inside trip_update.vehicle
        vehicle_id = None
        vehicle_label = None
        if tu.HasField("vehicle"):
            if tu.vehicle.HasField("id"):
                vehicle_id = tu.vehicle.id
            if tu.vehicle.HasField("label"):
                vehicle_label = tu.vehicle.label

        for stu in tu.stop_time_update:
            stop_sequence = stu.stop_sequence if stu.HasField("stop_sequence") else None
            stop_id = norm_stop_id(stu.stop_id) if stu.HasField("stop_id") else None

            rt_pred_arrival_ts = None
            rt_pred_departure_ts = None
            rt_arrival_delay_s = None

            if stu.HasField("arrival"):
                if stu.arrival.HasField("time"):
                    rt_pred_arrival_ts = int(stu.arrival.time)
                if stu.arrival.HasField("delay"):
                    rt_arrival_delay_s = int(stu.arrival.delay)

            if stu.HasField("departure"):
                if stu.departure.HasField("time"):
                    rt_pred_departure_ts = int(stu.departure.time)

            rows.append({
                "trip_id": trip_id,
                "route_id": route_id,
                "start_date": start_date,
                "start_time": start_time,
                "vehicle_id": vehicle_id,
                "vehicle_label": vehicle_label,
                "stop_sequence": stop_sequence,
                "stop_id": stop_id,
                "rt_pred_arrival_ts": rt_pred_arrival_ts,
                "rt_pred_departure_ts": rt_pred_departure_ts,
                "rt_arrival_delay_s": rt_arrival_delay_s,
            })

    df = pd.DataFrame(rows)
    return df
