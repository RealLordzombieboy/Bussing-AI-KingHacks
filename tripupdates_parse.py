from google.transit import gtfs_realtime_pb2
import pandas as pd
import re

def norm_stop_id(s: str) -> str:
    s = str(s).strip()
    # turn "00278" -> "S00278" to match your vehicle log normalization
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
    for e in feed.entity:
        if not e.HasField("trip_update"):
            continue

        tu = e.trip_update
        trip = tu.trip
        trip_id = trip.trip_id if trip.HasField("trip_id") else None

        start_date = trip.start_date if trip.HasField("start_date") else None
        start_time = trip.start_time if trip.HasField("start_time") else None
        route_id = trip.route_id if trip.HasField("route_id") else None

        for stu in tu.stop_time_update:
            stop_seq = stu.stop_sequence if stu.HasField("stop_sequence") else None
            stop_id = norm_stop_id(stu.stop_id) if stu.HasField("stop_id") else None

            arr_ts = stu.arrival.time if (stu.HasField("arrival") and stu.arrival.HasField("time")) else None
            dep_ts = stu.departure.time if (stu.HasField("departure") and stu.departure.HasField("time")) else None

            # delay is optional; arrival.delay is seconds vs schedule
            arr_delay = stu.arrival.delay if (stu.HasField("arrival") and stu.arrival.HasField("delay")) else None

            rows.append({
                "trip_id": trip_id,
                "route_id": str(route_id) if route_id is not None else None,
                "start_date": start_date,
                "start_time": start_time,
                "stop_sequence": stop_seq,
                "stop_id": stop_id,
                "rt_pred_arrival_ts": arr_ts,
                "rt_pred_departure_ts": dep_ts,
                "rt_arrival_delay_s": arr_delay,
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["trip_id", "stop_sequence", "stop_id"])
    df["trip_id"] = df["trip_id"].astype(str)
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce").astype("Int64")
    return df
