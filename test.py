import pandas as pd

df = pd.read_csv("vehicle_positions_log.csv")

usable = df.dropna(subset=["trip_id","route_id","stop_id","lat","lon"])
print("total rows:", len(df))
print("usable rows:", len(usable))
print("unique vehicles with usable rows:", usable["vehicle_id"].nunique())
print("unique trips:", usable["trip_id"].nunique())

usable = usable.sort_values(["vehicle_id","trip_id","feed_timestamp"])
changes = (usable.groupby(["vehicle_id","trip_id"])["stop_id"].shift(-1) != usable["stop_id"]) & usable.groupby(["vehicle_id","trip_id"])["stop_id"].shift(-1).notna()
print("stop transitions found:", changes.sum())