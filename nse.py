# import NSE_TIME_SERIES_ANALYSIS.nse as st
import streamlit as st  # For building the web app interface
import numpy as np
import tensorflow as tf
import pyarrow as pa
from datetime import datetime, timedelta
import plotly.graph_objects as go
from keras.models import load_model
from PIL import Image
import requests
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Kenyan flag
flag_url = "https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Kenya.svg"
# App config
st.set_page_config(page_title="Kenya Stock Market Dashboard", layout="wide")
# Header with flag
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    st.image(flag_url, width=80)

# Load the dataset
# Make sure the path to your CSV file is correct
df = pd.read_csv('NSE_2021_2024_Final.csv')

# Load your pre-trained model
# Make sure the path to your model file is correct
model = load_model('best_stacked_lstm_model.keras')

def recommend_stocks(investment_amount, risk_tolerance, investment_length, selected_sectors):
    """
    Recommends stocks based on user inputs.
    """
    # This is a simplified recommendation logic.
    # A more complex model would be needed for real-world financial advice.

    # Filter stocks based on some criteria (e.g., price for investment amount)
    max_price = investment_amount / 100 # Example: suggest stocks where 100 shares can be bought
    potential_stocks = df[df['Day Price'] <= max_price]

    # Filter by selected sectors if any are chosen
    if selected_sectors:
        potential_stocks = potential_stocks[potential_stocks['Sector'].isin(selected_sectors)]

    # Adjust recommendations based on risk tolerance
    if risk_tolerance == 'Low':
        # Prefer less volatile stocks (e.g., lower change percentage)
        potential_stocks = potential_stocks.sort_values(by='Change%', ascending=True)
    elif risk_tolerance == 'Medium':
        # Mix of stocks
        potential_stocks = potential_stocks.sample(frac=1).reset_index(drop=True)
    else: # High
        # Prefer more volatile stocks (e.g., higher change percentage)
        potential_stocks = potential_stocks.sort_values(by='Change%', ascending=False)

    # Adjust for investment length (simplified: longer length might favor growth stocks)
    if investment_length == 'Long-term (5+ years)':
        # You might prioritize stocks from certain sectors for long-term growth
        growth_sectors = ['Telecommunication & Technology', 'Financials', 'Banking'] # Example sectors
        long_term_stocks = potential_stocks[potential_stocks['Sector'].isin(growth_sectors)]
        if not long_term_stocks.empty:
            potential_stocks = long_term_stocks

    # Get top 5 recommendations
    recommendations = potential_stocks.head(5)

    return recommendations[['Name', 'Sector', 'Day Price']]


st.title('Stock Investment Recommender')

st.write("""
This app recommends stocks to invest in based on your financial preferences.
Please enter your investment amount, risk tolerance, and desired investment length.
Use the sidebar to filter by market sectors.
""")

# --- Sidebar for Sector Selection ---
st.sidebar.header('Market Sector Filters')
# Get unique sectors from the dataframe
sectors = sorted(df['Sector'].unique())
selected_sectors = st.sidebar.multiselect(
    'Select sectors you are interested in:',
    sectors
)


# --- Main Page for User Inputs ---
investment_amount = st.number_input('Enter your investment amount (in KES):', min_value=1000, max_value=10000000, value=50000, step=1000)
risk_tolerance = st.selectbox('Select your risk tolerance:', ('Low', 'Medium', 'High'))
investment_length = st.selectbox('Select your investment length:', ('Short-term (1-2 years)', 'Mid-term (3-5 years)', 'Long-term (5+ years)'))

if st.button('Get Recommendations'):
    with st.spinner('Generating recommendations...'):
        recommended_stocks = recommend_stocks(investment_amount, risk_tolerance, investment_length, selected_sectors)

        if not recommended_stocks.empty:
            st.success('Here are your top 5 stock recommendations:')
            st.table(recommended_stocks.reset_index(drop=True))
        else:
            st.warning('No stocks found matching your criteria. Please adjust your inputs or sector selection.')