# debug_overlap.py
# Quick diagnostic: compare IDs across VehiclePositions vs TripUpdates (LIVE)
# Uses repo file names seen in your screenshot.

import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from tripupdates_parse import parse_tripupdates_pb

VEH_LOG = "vehicle_positions_log.csv"
TRIPUPDATES_URL = "https://api.cityofkingston.ca/gtfs-realtime/tripupdates.pb"
TRIPUPDATES_TMP = "_tripupdates_live.pb"

TZ = ZoneInfo("America/Toronto")
UTC = ZoneInfo("UTC")


def fetch_tripupdates_live(out_path: str = TRIPUPDATES_TMP, timeout: int = 20) -> str:
    r = requests.get(TRIPUPDATES_URL, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def main():
    # 1) vehicle snapshot
    vdf = pd.read_csv(VEH_LOG)
    latest_ts = int(vdf["feed_timestamp"].max())
    snap = vdf[vdf["feed_timestamp"] == latest_ts].copy()

    snap_local = datetime.fromtimestamp(latest_ts, tz=UTC).astimezone(TZ)
    print(f"\nVehicle snapshot feed_timestamp={latest_ts} | local={snap_local}")

    snap = snap.dropna(subset=["vehicle_id", "route_id", "trip_id"]).copy()
    snap["vehicle_id"] = snap["vehicle_id"].astype(str)
    snap["route_id"] = snap["route_id"].astype(str)
    snap["trip_id"] = snap["trip_id"].astype(str)

    # 2) tripupdates live
    pb_path = fetch_tripupdates_live(TRIPUPDATES_TMP)
    tu = parse_tripupdates_pb(pb_path)

    # If your parser didn’t include route_id/trip_id for some reason
    for col in ["trip_id", "route_id", "stop_id"]:
        if col in tu.columns:
            tu[col] = tu[col].astype(str)

    print(f"\nTripUpdates rows: {len(tu)}")
    print("TripUpdates columns:", list(tu.columns))

    # 3) samples
    print("\n--- SAMPLE vehicle trip_ids ---")
    print(snap["trip_id"].head(10).to_list())

    if "trip_id" in tu.columns:
        print("\n--- SAMPLE tripupdates trip_ids ---")
        print(tu["trip_id"].dropna().head(10).to_list())
    else:
        print("\nTripUpdates has no trip_id column (parser issue).")

    # 4) overlap on trip_id
    if "trip_id" in tu.columns:
        overlap_trip = set(snap["trip_id"]).intersection(set(tu["trip_id"].dropna()))
        print(f"\nTrip ID overlap: {len(overlap_trip)}")
        if overlap_trip:
            print("Example overlapping trip_id:", next(iter(overlap_trip)))

    # 5) does tripupdates contain vehicle_id?
    if "vehicle_id" in tu.columns:
        tu_veh = tu.dropna(subset=["vehicle_id"]).copy()
        tu_veh["vehicle_id"] = tu_veh["vehicle_id"].astype(str)

        print("\n--- SAMPLE tripupdates vehicle_ids ---")
        print(tu_veh["vehicle_id"].head(10).to_list())

        overlap_vehicle = set(snap["vehicle_id"]).intersection(set(tu_veh["vehicle_id"]))
        print(f"\nVehicle ID overlap (vehicle feed vs tripupdates): {len(overlap_vehicle)}")
        if overlap_vehicle:
            print("Example overlapping vehicle_id:", next(iter(overlap_vehicle)))
    else:
        print("\nTripUpdates has NO vehicle_id column in parsed output (parser may not extract it).")

    # 6) route overlap sanity check
    if "route_id" in tu.columns:
        overlap_route = set(snap["route_id"]).intersection(set(tu["route_id"].dropna()))
        print(f"\nRoute overlap: {len(overlap_route)}")
        print("Vehicle routes sample:", snap["route_id"].dropna().unique()[:10])
        print("TripUpdates routes sample:", tu["route_id"].dropna().unique()[:10])

    print("\nDone.")


if __name__ == "__main__":
    main()
