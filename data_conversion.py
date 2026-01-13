from google.transit import gtfs_realtime_pb2
import requests

import csv
import io
from google.protobuf.json_format import MessageToDict

feed = gtfs_realtime_pb2.FeedMessage()

def get_current_data():
    response = requests.get("https://api.cityofkingston.ca/gtfs-realtime/vehicleupdates.pb")
    feed.ParseFromString(response.content)
    return feed.entity

#print(get_current_data()) # DEBUG

def proto_to_csv_string(repeated_container):
    """Converts a RepeatedCompositeContainer to a CSV string."""
    if not repeated_container:
        return ""

    # 1. Convert each proto message in the container to a Python dictionary
    #    The container is iterable.
    list_of_dicts = [MessageToDict(msg, preserving_proto_field_name=True) for msg in repeated_container]

    # Handle the case where the messages have different fields (not ideal for CSV)
    # For a simple, consistent schema:
    # 2. Extract field names (headers) from the first dictionary
    headers = list_of_dicts[0].keys()

    # 3. Write to a CSV format in memory using the `csv` module
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)

    writer.writeheader()
    writer.writerows(list_of_dicts)

    return output.getvalue()

def get_current_csv():
    csv_text = proto_to_csv_string(get_current_data())

    with open('current_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_text)
