# 📈 StockSense AI — Real-Time Stock Prediction Dashboard

A professional Streamlit app for real-time stock data visualization, technical analysis, and ML-based price prediction.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
stock_prediction/
│
├── app.py              ← Main Streamlit application (UI + charts)
├── data_fetcher.py     ← Yahoo Finance data fetching (cached)
├── predictor.py        ← ML models (Linear Regression, RF, LSTM, Ensemble)
├── utils.py            ← Technical indicators (RSI, MACD, Bollinger Bands, etc.)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## ✨ Features

### 📊 Charts
- **Candlestick chart** with real-time OHLCV data
- **Minute / Hour / Day / Week / Month** intervals
- Bollinger Bands overlay
- Moving Averages (MA5, MA10, MA20, MA50, MA200)
- Volume bars (color-coded by price direction)

### 🔮 AI Prediction
- **Linear Regression** — Fast baseline model
- **Random Forest** — Ensemble tree model (200 estimators)
- **LSTM Neural Network** — Deep learning (falls back to RF if TensorFlow not installed)
- **All Models Ensemble** — Averages predictions from all models
- Confidence score, RMSE, MAE, R², MAPE metrics
- Day-by-day forecast table with BUY/SELL signals

### 📉 Technical Indicators
- RSI (14) with overbought/oversold zones
- MACD with histogram
- Stochastic Oscillator (%K, %D)
- ATR, OBV

### 🏢 Company Support
- 19 popular pre-loaded companies (AAPL, MSFT, GOOGL, TSLA, NVDA, etc.)
- Custom ticker input (supports NSE India: `.NS`, BSE: `.BO`)

---

## ⚙️ Configuration

| Setting | Options |
|---------|---------|
| Intervals | 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo |
| Period | 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y |
| Prediction | 1–30 days ahead |
| Auto Refresh | Every 60 seconds |

---

## ⚠️ Disclaimer
This app is for **educational purposes only**. Predictions are not financial advice. Past performance does not guarantee future results.

---

## 🛠️ Optional: Enable LSTM (Deep Learning)
```bash
pip install tensorflow
```
If not installed, LSTM automatically falls back to Random Forest.
