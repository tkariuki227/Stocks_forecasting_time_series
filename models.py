from prophet import Prophet
import pandas as pd
import os

features = ['Day Price', '30_day_SMA', '30_day_EMA', '100_day_SMA', '100_day_EMA']

merged_df = pd.read_pickle('merged_nse_df.pkl')

def predict_prophet(merged_df):
    df_prophet = merged_df[['Date', 'Day Price']].rename(columns={'Date':'ds', 'Day Price':'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return forecast

def predict_with_lstm(merged_df, features=features, target_col = 'Day Price', lookback_days=100, epochs=50, future_days=0, save_model=False):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    import pandas as pd

    #  Split into train/test (80%/20%)
    data = merged_df[['Date'] + features].copy()
    target_data = data[features].values
    target_idx = features.index(target_col)


    train_size = int(len(data) * 0.8)
    data_training = target_data[:train_size]
    data_testing  = target_data[train_size:]

    #  Scale
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_scaled = scaler.fit_transform(data_training)

    #  Training sequences
    X_train, y_train = [], []
    for i in range(lookback_days, len(data_training_scaled)):
        X_train.append(data_training_scaled[i-lookback_days:i])
        y_train.append(data_training_scaled[i, target_idx])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model_path = "lstm_final_model.h5"
    if os.path.exists(model_path) and not save_model:
        print("Loading existing LSTM model...")
        from keras.models import load_model
        model = load_model(model_path)
    else:

    #  LSTM model
        model = Sequential()
        model.add(LSTM(units=70, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=90, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        #  Train
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        #  Save model after training (optional)
        if save_model:
            model.save(model_path)
            print("Model saved as lstm_model.h5")

    #  Test data (last 100 days + test set)
    past_days = data_training[-lookback_days:]
    final_test_input = np.concatenate((past_days, data_testing), axis=0)
    final_test_scaled = scaler.transform(final_test_input)

    X_test, y_test = [], []
    for i in range(lookback_days, len(final_test_scaled)):
        X_test.append(final_test_scaled[i-lookback_days:i])
        y_test.append(final_test_scaled[i, target_idx])
    X_test, y_test = np.array(X_test), np.array(y_test)

    #  Predict
    y_pred_scaled = model.predict(X_test)

    # Rescale ONLY the target column
    dummy = np.zeros((len(y_pred_scaled), len(features))) # same feature count
    dummy[:, target_idx] = y_pred_scaled[:, 0]
    y_pred = scaler.inverse_transform(dummy)[:, target_idx]

    dummy_test = np.zeros((len(y_test), len(features)))
    dummy_test[:, target_idx] = y_test
    y_test_real = scaler.inverse_transform(dummy_test)[:, target_idx]
    # y_pred = scaler.inverse_transform(y_pred_scaled)

    #  Results DataFrame
    test_dates = data['Date'].iloc[-len(y_test):].values
    result_df = pd.DataFrame({
        'date': test_dates,
        'actual': y_test_real,
        'predicted': y_pred
    })

     # FUTURE FORECAST: predict next N days autoregressively
    future_inputs = final_test_scaled[-lookback_days:]  # last known window
    future_preds = []
    for _ in range(future_days):
        X_future = future_inputs[-lookback_days:].reshape(1, lookback_days, len(features))
        pred_scaled = model.predict(X_future)
        future_preds.append(pred_scaled[0][0])

        new_row = np.zeros(len(features))  # dummy for future
        new_row[target_idx] = pred_scaled[0][0]
        future_inputs = np.vstack([future_inputs, new_row])
        # future_inputs = np.append(future_inputs, pred_scaled, axis=0)
    

    dummy_future = np.zeros((len(future_preds), len(features)))
    dummy_future[:, target_idx] = future_preds
    future_prices = scaler.inverse_transform(dummy_future)[:, target_idx]

    # Generate future dates
    last_date = merged_df['Date'].iloc[-1]
    future_dates = pd.date_range(last_date, periods=future_days+1, freq='B')[1:]
    future_df = pd.DataFrame({'date': future_dates, 'predicted': future_prices})

    # future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
    # future_dates = pd.date_range(merged_df['Date'].iloc[-1], periods=future_days+1, freq='B')[1:]
    # future_df = pd.DataFrame({'date': future_dates, 'predicted': future_prices})

    return result_df, future_df

    # return result_df
