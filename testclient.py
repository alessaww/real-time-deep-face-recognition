from jsonsocket import Client, Server
import sys

host = 'localhost'
port = 10003

data = {
    'getid': 1,
    'path': 'person1_d38.jpg'
}
print(sys.stderr, 'Agent 4 sends "%s"' % data)
# or in one line:
response = Client().connect(host, port).send(data).recv()
print(sys.stderr, 'Agent 4 receives "%s"' % response)
