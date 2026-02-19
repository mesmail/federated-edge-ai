import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate sample IoT sensor data
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.normal(5, 1, (50, 2))

data = np.vstack([normal_data, anomaly_data])

# Train anomaly detection model
model = IsolationForest(contamination=0.05)
model.fit(data)

# Predict anomalies
predictions = model.predict(data)

# Plot results
plt.scatter(data[:,0], data[:,1], c=predictions)
plt.title("Edge AI Anomaly Detection")
plt.show()

print("Model trained successfully")
