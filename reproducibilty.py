"""
Reproducibility Script for NSE Stock Recommender
----------------------------------------------------
This script loads the NSE dataset, preprocesses it, loads the trained models,
makes predictions, and saves the recommendation results.

Author: Daniel Mutiso, Sylvia Mwangi, Stephen Kamiru, Teresia Kariuki, Meggy Ataro
Version: 0.1.0
Date: 2025-07
"""

# 📦 Dependencies
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from keras.models import load_model

# 📁 File Paths
DATA_PATH = "NSE_2021_2024_Final.csv"
XGB_MODEL_PATH = "best_xgb_model.pkl"
LSTM_MODEL_PATH = "best_stacked_lstm_model.keras"
OUTPUT_PATH = "stock_recommendations.csv"

# 1️⃣ Load Dataset
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# 2️⃣ Basic Preprocessing
print("[INFO] Preprocessing dataset...")
df = df.dropna(subset=["Volume", "Change%", "Day Price"])  # remove rows with essential missing values
df["Change%"] = df["Change%"].str.replace('%', '').astype(float)
df["Volume"] = df["Volume"].str.replace(',', '').astype(float)

# Select features for XGBoost
features = ["Volume", "Day Price", "Change%"]
X = df[features]

# 3️⃣ Feature Scaling
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Load XGBoost Model
print("[INFO] Loading XGBoost model...")
xgb_model = joblib.load(XGB_MODEL_PATH)

# 5️⃣ Make Predictions
print("[INFO] Making predictions...")
predictions = xgb_model.predict(X_scaled)
df["Prediction"] = predictions
df["Recommendation"] = df["Prediction"].apply(lambda x: "BUY" if x == 1 else "IGNORE")

# 6️⃣ Top 5 Recommendations
recommended = df[df["Prediction"] == 1].sort_values(by="Change%", ascending=False)
top5 = recommended.head(5)[["Name", "Code", "Day Price", "Change%", "Volume", "Recommendation"]]

# 7️⃣ Save Output
print(f"[INFO] Saving top 5 recommendations to {OUTPUT_PATH}...")
top5.to_csv(OUTPUT_PATH, index=False)

print("[✅ DONE] Reproducibility script completed.")
