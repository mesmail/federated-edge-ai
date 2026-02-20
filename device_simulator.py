import requests
import random
import time

SERVER_URL = "http://127.0.0.1:8000/predict"

def generate_sensor_data():

    # simulate normal data
    if random.random() < 0.8:
        x = random.normalvariate(0, 1)
        y = random.normalvariate(0, 1)

    # simulate anomaly
    else:
        x = random.normalvariate(5, 1)
        y = random.normalvariate(5, 1)

    return x, y


while True:

    x, y = generate_sensor_data()

    response = requests.post(
        SERVER_URL,
        params={"x": x, "y": y}
    )

    result = response.json()

    print(f"Sensor data: x={x:.2f}, y={y:.2f} → {result['result']}")

    time.sleep(2)