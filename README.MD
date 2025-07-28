# 📊 Nairobi Stock Exchange (NSE) Time Series Analysis

The objective of this project was to build a machine learning–powered web application that recommends promising stocks on the Nairobi Securities Exchange (NSE). By leveraging real historical data (2021–2024), the tool helps investors—especially in the Kenyan retail sector—make data-informed decisions about where to invest.

![image](Images/image_3.jpg)

## Table of Contents

- [Overview](#overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)


## 📝 Project Overview

This project focuses on analyzing and forecasting stock market trends using time series data from the *Nairobi Stock Exchange (NSE)*. We use historical data from 2021 to identify stock price trends, sector performance, and volatility patterns. We also forecast future prices using advanced modeling techniques.

The goal is to provide valuable insights for investors, analysts, and policymakers using data science.

---

## Business Understanding

The project aims to empower Kenyan retail investors with a reliable decision-support tool for identifying top-performing stocks. By analyzing historical and current stock data and applying deep learning, the system provides actionable insights based on user-defined investment preferences to help them make smarter and more confident decisions.
![Image](Images/image_2.jpg)

## 🎯 Objectives

- Analyze historical stock price movements
- Compare sector-level performance
- Study daily returns and volatility
- Forecast future stock prices using time series models

---

## *Business Goals*

- Recommend the top 5 performing stocks based on historical and real-time data.

- Provide a user-friendly interface (Streamlit) to interact with filtered data and generate insights.

- Assist investors with risk-aware investment decisions through precise predictions.

---

## 📁 Dataset Understanding

- *Source:* Scraped from the official Nairobi Stock Exchange website.  [here](https://www.nse.co.ke/dataservices/)
- *Time Period:* 2021 (Daily frequency)  
- *Rows:* 64,688  
- *Key Columns:*  
  - Date  
  - Code (Stock symbol)  
  - Name (Company name)  
  - Day High, Day Low, Day Price  
  - Previous Price, Change, Change %  
  - Volume, Adjusted Price  

---

## 🧹 Data Preparation

- Cleaned and formatted numerical columns  
- Converted date strings to datetime format  
- Removed duplicates and handled missing values  
- Created time series per stock for trend analysis and modeling
- Use of pipelines for preprocessing.

---

## 🔍 Exploratory Data Analysis (EDA)

- Visualized stock price movements over time  
- Analyzed daily returns to identify volatile stocks  
- Grouped and compared performance by sector  
- Highlighted key trends in top stocks like *Safaricom* and *Equity Bank*

---

## 🧠 Forecasting Models Used

### 1. *ARIMA*  

Statistical model that uses past values and lagged differences to make forecasts. Suitable for linear, stationary data.

### 2. *Prophet*

Model by Facebook for easy, interpretable time series forecasting. Captures trends, seasonality, and holidays.

### 3. *LSTM (Long Short-Term Memory)*  

A deep learning model ideal for time-dependent data. Captures non-linear, complex patterns using memory cells.  
Used to forecast Safaricom stock prices, showing good ability to follow actual trends.

---

## Evaluation

- **Metrics:** RMSE (Root Mean Squared Error) for prediction accuracy.
- **Validation:** Comparing models using cross-validation and visualizing user-level predictions.
- **Selection:** Choosing the best-performing model based on accuracy and robustness.

## Model Deployment

The trained recommendation model is deployed as an interactive web application using [Streamlit](https://streamlit.io/). Users can input their preferences and receive personalized movie recommendations in real time.


- **Access:** Run the Streamlit app locally with:
  
  ```bash
  streamlit run app/nse.py
  ```

---

| File                             | Purpose                                      |
|----------------------------------|----------------------------------------------|
| `nse.py`                         | Main Streamlit application                   |
| `NSE_2021_2024_Final.csv`        | Cleaned historical stock data                |
| `best_stacked_lstm_model.keras`  | Trained Keras LSTM model                     |
| `EDA.ipynb`| Notebook with cell-by-cell summaries         |
| `requirements.txt`               | Environment dependencies                     |

---

- **Features:** 
  - User-friendly interface for entering ratings or selecting favorite genres.
  - Displays top recommended movies instantly.
  - Visualizations of recommendation results and user history.

---

- **Notebook Usage:** All steps are reproducible in `notebooks/EDA.ipynb and Model.ipynb`.

- **Instructions:**
  1. Download the MovieLens dataset and place `ratings.csv` and `movies.csv` in the `data/` folder.
  2. Install dependencies:

      ```bash
      pip install -r requirements.txt
      ```

  3. Launch the notebook:

      ```bash
      jupyter notebook notebooks/EDA.ipynb & model.ipynb
      ```

  4. Follow the notebook steps: Data Understanding → Preparation → Modeling → Evaluation → Deployment

---

## 📈 Key Insights

- Safaricom showed consistent long-term growth  
- Banking sector was highly volatile  
- LSTM performed better than traditional models for capturing stock price behavior  
- Sector-level comparison revealed market imbalances and growth opportunities


## 📈 Next Steps / Future Improvements

Integrate live NSE API or automatic data refresh
Enhance forecasting with improved LSTM input pipelines
Add portfolio builder or backtesting tools
Build a mobile-ready UI or Android wrapper for retail investors

## Credits
- [Sylvia Mwangi](https://github.com/Sylvia-W-Mwangi)
- [Daniel Mutiso](https://github.com/dantegaucho)
- [Stephen Kamiru](https://github.com/Kamiru17)
- [Teresia Njango](https://github.com/tkariuki227)
- [Meggy Donna](https://github.com/MegAtaro)

---

## 📢 Disclaimer

This is a demonstration project. It is not intended to provide financial advice. Please consult a licensed financial advisor before making any investment decisions.

## LICENSE

This project is for educational and academic use. Attribution required if reused.
