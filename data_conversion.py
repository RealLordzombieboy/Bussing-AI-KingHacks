from google.transit import gtfs_realtime_pb2
import requests

import csv
import time

feed = gtfs_realtime_pb2.FeedMessage()

def get_current_data():
    response = requests.get("https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb")
    feed.ParseFromString(response.content)
    return feed.entity

def proto_to_csv_string(current_data=get_current_data()):
    csv_dict = [["trip_id", "route_id", "stop_id", "speed_mps", "feed_timestamp", "vehicle_id"]]
    for entity in current_data:
        #print(entity) # DEBUG
        # If the bus is on a route.
        if (entity.vehicle.HasField("trip")):
            row = [f"{entity.vehicle.trip.trip_id}", f"{entity.vehicle.trip.route_id}", f"{entity.vehicle.stop_id}", f"{entity.vehicle.position.speed}", f"{entity.vehicle.trip.start_time}", f"{entity.vehicle.vehicle.id}"]
            csv_dict.append(row)

    return csv_dict


#print(get_current_data()) # DEBUG
#print(proto_to_csv_string(get_current_data())) # DEBUG

if __name__ =="__main__":
    n = 1
    # Started at 9:46 AM January 15th, 2026.
    while True:
        csv_dict = proto_to_csv_string(get_current_data())
        
        with open(f'Jan 15 data/current_data{n}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_dict)
        n += 1
        time.sleep(10)