# 📈 DeepStock: Google Price Forecasting


## 🚀 Project Overview
This repository contains an **LSTM-based deep learning model** for forecasting **Google stock prices** using historical data. The model leverages **sequential dependencies** in stock prices to make predictions and evaluate performance using RMSE.

## 📊 Model Performance
- **Root Mean Squared Error (RMSE):** `0.0241562`
- The model effectively captures the **trend** of Google's stock price, though improvements can be made in handling sudden fluctuations.
- The comparison between **actual vs. predicted stock prices** is visualized in the plot above.

## 🏗️ Model Architecture
The model consists of:
- **4 LSTM layers** with `return_sequences=True` (except the last one).
- **Dropout layers (0.2 probability)** to prevent overfitting.
- **A Dense output layer** with `units=1` for price prediction.

### 📌 LSTM Model Summary:
```python
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # Final output layer

```
📌 Results & Observations
The model provides a reasonably accurate stock price prediction with a low RMSE of 0.0241562.

The actual vs. predicted price plot shows the model successfully follows the overall trend.

The model can be further improved by:

Adding technical indicators (e.g., RSI, MACD).

Using Bidirectional LSTMs for better sequence learning.

Training on larger datasets with more features.
