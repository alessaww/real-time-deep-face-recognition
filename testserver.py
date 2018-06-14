from jsonsocket import Client, Server
import sys

host = 'localhost'
port = 10000

server = Server(host, port)
while True:
    data = server.accept().recv()
    print(sys.stderr, 'Kinect receives "%s"' % data)
    send_data = {'confidence': 1.0, 'scared': 0.3, 'sad': 0.0, 'neutral': 0.0, 'surprise': 0.0, 'bored': 0.0, 'angry': 0.0, 'happy': 1.0}
    print(sys.stderr, 'Kinect sends "%s"' % send_data)
    server.send(send_data)