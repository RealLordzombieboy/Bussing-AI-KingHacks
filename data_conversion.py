with open("vehicleupdates.pb", "rb") as f:
    binary_data = f.read()

from google.transit import gtfs_realtime_pb2
import requests

feed = gtfs_realtime_pb2.FeedMessage()
response = requests.get("https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb")
feed.ParseFromString(response.content)
for entity in feed.entity:
    print(entity)