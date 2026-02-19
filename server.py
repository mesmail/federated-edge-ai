import numpy as np
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI
import uvicorn

# Create FastAPI app
app = FastAPI()

# Train model once
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.normal(5, 1, (50, 2))

data = np.vstack([normal_data, anomaly_data])

model = IsolationForest(contamination=0.05)
model.fit(data)

@app.get("/")
def root():
    return {"message": "Edge AI server is running"}

@app.post("/predict")
def predict(x: float, y: float):

    input_data = np.array([[x, y]])

    prediction = model.predict(input_data)

    if prediction[0] == -1:
        result = "anomaly"
    else:
        result = "normal"

    return {
        "x": x,
        "y": y,
        "result": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
