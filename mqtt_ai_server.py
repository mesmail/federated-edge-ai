import json
import numpy as np
from sklearn.ensemble import IsolationForest
import paho.mqtt.client as mqtt

# Train anomaly detection model
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.normal(5, 1, (50, 2))
data = np.vstack([normal_data, anomaly_data])

model = IsolationForest(contamination=0.05)
model.fit(data)

# MQTT settings
BROKER = "localhost"
PORT = 1883
TOPIC = "iot/sensor"

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker successfully")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())

    x = payload["x"]
    y = payload["y"]

    prediction = model.predict([[x, y]])

    result = "ANOMALY" if prediction[0] == -1 else "NORMAL"

    print(f"Received: x={x:.2f}, y={y:.2f} → {result}")

client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)

client.loop_forever()