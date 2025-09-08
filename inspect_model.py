import os
import joblib
import pandas as pd
import numpy as np

# ------------------ Detect project root ------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------ Model path ------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "scripts", "delay_predictor.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_ml.py first.")

# ------------------ Load model ------------------
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
encoders = model_data["encoders"]

print("✅ Model and encoders loaded successfully")
print("Model keys:", model_data.keys())
print("Encoders:", list(encoders.keys()))

# ------------------ Sample input ------------------
sample_input = {
    "Train_Type": "Express",           # Must exist in training data ideally
    "Passenger_Load": "High",
    "Distance": 250,
    "Weather": "Rainy",
    "Congestion_Level": "Medium",      # If unseen, will be handled
    "Maintenance_Risk": "Low"
}

df = pd.DataFrame([sample_input])

# ------------------ Safe Encoding ------------------
for col, le in encoders.items():
    if col in df.columns:
        # Map unseen labels to -1
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else None)
        df[col] = le.transform(df[col].fillna("__UNKNOWN__")) if "__UNKNOWN__" in le.classes_ else le.transform(df[col].fillna(le.classes_[0]))
    else:
        raise ValueError(f"Missing required column: {col}")

# ------------------ Predict ------------------
predicted_delay = model.predict(df)
print(f"\n🚂 Sample Prediction: {predicted_delay[0]:.2f} mins delay")
