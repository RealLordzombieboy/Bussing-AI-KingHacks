Use "protoc --proto_path=. --python_out=. gtfs-realtime.proto" in a terminal cd'd into this folder to create a python converter for .pb to .py of the Kingston Transit .pb files.
To install protoc: https://protobuf.dev/installation/

To run the demo follow the steps: 

1. Run data_conversion.py in one terminal

2. Run live_api.py in another terminal

3. In a third terminal: run python -m http.server 8000 on a third terminal to launch the site!

Stay Tuned, in the future coming to a real app!

