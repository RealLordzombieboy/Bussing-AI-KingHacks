import gtfs_realtime_pb2

message = gtfs_realtime_pb2.DESCRIPTOR

# Read data from .pb file
with open("vehicleupdates.pb", "rb") as f:
    binary_data = f.read()
    print(binary_data)