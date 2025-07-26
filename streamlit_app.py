# # import streamlit as st
# # import pandas as pd


# # st.title("📈 NSE Stock Prediction App")

# # # Load the already cleaned data
# # merged_df = pd.read_pickle("merged_nse_df.pkl")

# # # Show available stocks
# # stock_list = sorted(merged_df['Name'].unique())
# # selected_stock = st.selectbox("Choose a stock", stock_list)

# # # Filter and display
# # df_stock = merged_df[merged_df['names'] == selected_stock].copy()
# # st.line_chart(df_stock.set_index('date')['day price'])


# # df_stock = get_stock_timeseries(merged_df, selected_stock)
# # st.line_chart(df_stock.set_index('date')['close'])

# # # Prophet Prediction
# # if st.button("Predict Next 30 Days"):
# #     forecast = predict_with_prophet(df_stock, future_days=30)
# #     st.write(forecast[['ds','yhat']].tail())
# #     st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])


# import streamlit as st
# import pandas as pd
# from models import predict_prophet, predict_with_lstm
# from individual_stock import get_stock 

# st.title("📈 NSE Stock Prediction App")

# merged_df = pd.read_pickle('merged_nse_df.pkl')

# # Stock dropdown
# stock_list = sorted(merged_df['Name'].unique())
# selected_stock = st.selectbox("Choose a stock", stock_list)

# # Show its time series
# df_stock = get_stock(merged_df, selected_stock)
# st.line_chart(df_stock.set_index('Date')['Day Price'])

# # Model selection
# model_choice = st.selectbox("Select a prediction model", ["Prophet", "LSTM"])

# # Prediction button
# if st.button("Predict Next 30 Days"):
#     if model_choice == "Prophet":
#         forecast = predict_prophet(df_stock)
#         st.write(forecast[['ds','yhat']].tail())
#         st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
    
#     elif model_choice == "LSTM":
#         # forecast = predict_with_lstm(df_stock, future_days=30)
#         # st.write(forecast.tail())
#         # st.write("Forecast columns:", forecast.columns)
#         # st.write(forecast.head())

#         # st.line_chart(forecast.set_index('date')['predicted'])

#         backtest_df, future_df = predict_with_lstm(df_stock, future_days=30)

#         backtest_df = backtest_df.dropna()

#         from sklearn.metrics import mean_absolute_error, mean_squared_error
#         import numpy as np

#         mae = mean_absolute_error(backtest_df['actual'], backtest_df['predicted'])
#         rmse = np.sqrt(mean_squared_error(backtest_df['actual'], backtest_df['predicted']))
#         mape = np.mean(np.abs((backtest_df['actual'] - backtest_df['predicted']) / backtest_df['actual'])) * 100

#         st.write(f"📊 **Backtest Metrics**")
#         st.write(f"- MAE: {mae:.2f}")
#         st.write(f"- RMSE: {rmse:.2f}")
#         st.write(f"- MAPE: {mape:.2f}%")


#         # Show backtest (last known 30 days)
#         st.subheader("✅ LSTM Backtest (last 30 days)")
#         st.dataframe(backtest_df.tail(30))
#         st.line_chart(backtest_df.set_index('date')[['actual', 'predicted']])

#         # Show future forecast if available
#         if not future_df.empty:
#             st.subheader("📅 LSTM Future Predictions (next 30 days)")
#             st.dataframe(future_df)
#             st.line_chart(future_df.set_index('date')['predicted'])
#         else:
#             st.info("ℹ️ No future forecast requested.")


# if model_choice == "Prophet":
#     forecast = predict_prophet(df_stock)
#     st.write("✅ Prophet forecast")
#     st.write(forecast.tail())  # Works because Prophet returns a DataFrame
#     st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])

# elif model_choice == "LSTM":
#     backtest_df, future_df = predict_with_lstm(df_stock, lookback_days=100, epochs=10, future_days=30)
    
#     st.write("✅ LSTM Backtest (last 30 days actual vs predicted)")
#     st.write(backtest_df.tail(30))  # last 30 known days
    
#     st.write("✅ LSTM Future Predictions")
#     st.write(future_df.tail())  # future predictions
    
#     # Show backtest chart
#     st.line_chart(backtest_df.set_index('date')[['actual', 'predicted']])
    
#     # Show future predictions chart
#     st.line_chart(future_df.set_index('date')['predicted'])


import streamlit as st
import pandas as pd
from models import predict_prophet, predict_with_lstm
from individual_stock import get_stock
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


st.title("📈 NSE Stock Prediction App")

# Load data
merged_df = pd.read_pickle('merged_nse_df.pkl')

# Stock dropdown
stock_list = sorted(merged_df['Name'].unique())
selected_stock = st.selectbox("Choose a stock", stock_list)

# Filter selected stock
df_stock = get_stock(merged_df, selected_stock)
st.line_chart(df_stock.set_index('Date')['Day Price'])

# Model selection
model_choice = st.selectbox("Select a prediction model", ["Prophet", "LSTM"])

if model_choice == "LSTM":
    # Let user choose model type
    model_type = st.radio(
        "Which LSTM model do you want?",
        ["Base model (Day Price only)", "Enhanced model (with SMAs/EMAs)"]
    )

    # ✅ Decide features based on choice
    if model_type == "Base model (Day Price only)":
        features = ["Day Price"]
    else:
        features = [
            "Day Price",
            "30_day_SMA",
            "30_day_EMA",
            "100_day_SMA",
            "100_day_EMA"
        ]

# Prediction button
if st.button("Predict Next 30 Days"):
    if model_choice == "Prophet":
        forecast = predict_prophet(df_stock)
        st.write(forecast[['ds','yhat']].tail())
        st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
    
    elif model_choice == "LSTM":
        backtest_df, future_df = predict_with_lstm(df_stock, features, future_days=30)

        # Drop NA in case of rolling window gaps
        backtest_df = backtest_df.dropna()

        # Compute metrics
        mae = mean_absolute_error(backtest_df['actual'], backtest_df['predicted'])
        rmse = np.sqrt(mean_squared_error(backtest_df['actual'], backtest_df['predicted']))
        mape = np.mean(np.abs((backtest_df['actual'] - backtest_df['predicted']) / backtest_df['actual'])) * 100

        st.write(f"📊 **Backtest Metrics**")
        st.write(f"- MAE: {mae:.2f}")
        st.write(f"- RMSE: {rmse:.2f}")
        st.write(f"- MAPE: {mape:.2f}%")

        # Backtest display
        st.subheader("✅ LSTM Backtest (last 30 days)")
        st.dataframe(backtest_df.tail(30))
        st.line_chart(backtest_df.set_index('date')[['actual', 'predicted']])

        # Future forecast
        if not future_df.empty:
            st.subheader("📅 LSTM Future Predictions (next 30 days)")
            st.dataframe(future_df)
            st.line_chart(future_df.set_index('date')['predicted'])
        else:
            st.info("ℹ️ No future forecast requested.")
