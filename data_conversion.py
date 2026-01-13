from google.transit import gtfs_realtime_pb2
import requests

feed = gtfs_realtime_pb2.FeedMessage()

def get_current_data():
    response = requests.get("https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb")
    return feed.ParseFromString(response.content)