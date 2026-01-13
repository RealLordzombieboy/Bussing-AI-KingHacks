from google.transit import gtfs_realtime_pb2
import requests

import csv

feed = gtfs_realtime_pb2.FeedMessage()

def get_current_data():
    response = requests.get("https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb")
    feed.ParseFromString(response.content)
    return feed.entity

def proto_to_csv_string(current_data):
    csv_dict = [["trip_id", "route_id", "stop_id", "speed_mps", "feed_timestamp", "vehicle_id"]]
    for entity in current_data:
        # If the bus is on a route.
        #print(entity) # DEBUG
        if (entity.vehicle.HasField("trip")):
            row = [f"{entity.vehicle.trip.trip_id}", f"{entity.vehicle.trip.route_id}", f"{entity.vehicle.stop_id}", f"{entity.vehicle.position.speed}", f"{entity.vehicle.trip.start_time}", f"{entity.vehicle.vehicle.id}"]
            
            csv_dict.append(row)
    print(csv_dict)

    with open('current_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_dict)

#print(get_current_data()) # DEBUG
#print(proto_to_csv_string(get_current_data())) # DEBUG
