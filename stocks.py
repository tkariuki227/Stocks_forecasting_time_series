import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# === Load Model and Data ===
model = joblib.load("best_xgb_model.pkl")
df = pd.read_csv("NSE_2021_2024_Final.csv")
df["Date"] = pd.to_datetime(df["Date"])

# === Streamlit UI Config ===
st.set_page_config("📈 NSE Stock Recommender", layout="wide")

# === Sidebar Filters ===
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Kenya.svg", width=100)
st.sidebar.title("🔍 Filter Options")

available_sectors = sorted(df["Sector"].dropna().unique())
selected_sector = st.sidebar.selectbox("📂 Select Sector", ["All"] + available_sectors)

selected_date = st.sidebar.date_input("📅 Select Date", value=df["Date"].max())
min_volume = st.sidebar.slider("📦 Minimum Volume", min_value=0, max_value=1000000, value=10000, step=1000)

# === Data Filtering ===
filtered_df = df[df["Date"] == pd.to_datetime(selected_date)]
filtered_df = filtered_df[filtered_df["Volume"] >= min_volume]
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df["Sector"] == selected_sector]

# Group by latest stock info (assumes one row per stock on a day)
stock_data = filtered_df.drop_duplicates(subset=["Code"], keep="last")

# Ensure required columns exist
required_features = ["Day Price", "12m Low", "12m High", "Day Low", "Day High", "Volume", "Change", "Change%"]
missing_cols = [col for col in required_features if col not in stock_data.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# === Feature Preparation ===
X = stock_data[required_features]
X = X.fillna(0)  # Handle missing values

# === Model Prediction ===
stock_data["Score"] = model.predict_proba(X)[:, 1]  # probability of class 1
top_stocks = stock_data.sort_values("Score", ascending=False).head(5)

# === Main App UI ===
st.title("Kenyan Stock Recommender")
st.markdown("📊 Smart suggestions for what to buy on the NSE based on your filters.")

if top_stocks.empty:
    st.warning("No stocks match your filters.")
else:
    for _, row in top_stocks.iterrows():
        name = row["Name"]
        code = row["Code"]
        score = row["Score"]
        ticker = f"{code}.NR"  # Simulated NSE Yahoo Finance ticker format

        st.markdown(f"## 📌 {name} ({code}) — Score: `{score:.2f}`")

        try:
            data = df.download(ticker, period="1mo", interval="1d", progress=False)
            if data.empty:
                raise ValueError("No data returned")

            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            fig.update_layout(title=f"{name} - 30 Day Trend", height=300, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Could not fetch chart for {name} ({ticker})")

st.markdown("---")
st.markdown("<small>Made with ❤️ using AI and the Nairobi Securities Exchange data.</small>", unsafe_allow_html=True)
