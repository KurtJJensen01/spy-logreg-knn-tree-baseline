# SPY Next-Day Direction Prediction (ML)

Predict whether the SPDR S&P 500 ETF (**SPY**) will close higher the next trading day using classic machine learning models and simple technical features.

This project:
- Downloads historical SPY OHLCV data from Kaggle using `kagglehub`
- Engineers technical features (log returns, SMAs, rolling returns, volatility)
- Uses a **time-based** train/validation/test split (70% / 10% / 20%)
- Trains and evaluates:
  - Logistic Regression (with scaling)
  - K-Nearest Neighbors (with scaling)
  - Decision Tree
- Compares against a baseline model (**always predict “up”**)
- Produces plots:
  - Distribution of daily log returns
  - Test F1 score comparison across models
- Prints a simple “helper output” for the most recent day in the dataset using Logistic Regression

> ⚠️ Educational project — not financial advice. This is a toy example and does not account for transaction costs, slippage, regime changes, or proper walk-forward tuning.

---

## Dataset

The script downloads this Kaggle dataset via `kagglehub`:

- **aliplayer1/historical-price-and-volume-data-for-spy-etf**
- File used: `SPY_prices.csv`

---

## Features

Engineered features include:
- **Return_1d**: daily log return  
- **SMA_5**, **SMA_20**: simple moving averages  
- **Return_5d**, **Return_20d**: rolling summed log returns  
- **Volatility_5d**, **Volatility_20d**: rolling std of log returns  
- Plus raw OHLCV: Open, High, Low, Close, Volume

Target label:
- **UpTomorrow** = 1 if `Close(t+1) > Close(t)`, else 0

---

## Train / Validation / Test split (time-based)

The dataset is sorted by date and split chronologically:
- Train: first 70%
- Val: next 10%
- Test: final 20%

This avoids leakage from shuffling time series data.

---

## Metrics

For each model (and the baseline), the script reports:
- Accuracy
- F1 score (for the “Up” class)

---

## How to run

### 1) Create environment & install dependencies

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib
