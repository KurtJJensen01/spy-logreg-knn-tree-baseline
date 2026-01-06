import kagglehub
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------

path = kagglehub.dataset_download("aliplayer1/historical-price-and-volume-data-for-spy-etf")
csv_path = os.path.join(path, "SPY_prices.csv")

df = pd.read_csv(csv_path)

# Parse Date and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# -----------------------------
# 2. Create target + features
# -----------------------------

# Next-day close and binary label
df['Close_tomorrow'] = df['Close'].shift(-1)
df['UpTomorrow'] = (df['Close_tomorrow'] > df['Close']).astype(int)
df = df.iloc[:-1].copy()  # drop last row without tomorrow

# Technical features
df['Return_1d'] = np.log(df['Close'] / df['Close'].shift(1))

df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()

df['Return_5d'] = df['Return_1d'].rolling(window=5).sum()
df['Return_20d'] = df['Return_1d'].rolling(window=20).sum()

df['Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()

df = df.dropna().reset_index(drop=True)

feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Return_1d', 'SMA_5', 'SMA_20',
    'Return_5d', 'Return_20d',
    'Volatility_5d', 'Volatility_20d'
]

X = df[feature_cols].copy()
y = df['UpTomorrow'].copy()

print(f"Total samples after feature engineering: {len(df)}")
print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
print()


# ---- Figure 1: distribution of daily log returns ----
plt.figure()
df['Return_1d'].hist(bins=50)
plt.xlabel("Daily log return")
plt.ylabel("Frequency")
plt.title("Distribution of daily SPY daily log returns")
plt.tight_layout()


# -----------------------------
# 3. Time-based train / val / test split by percentage
# -----------------------------

n = len(df)

train_end_idx = int(0.70 * n)   # first 70% for training
val_end_idx   = int(0.80 * n)   # next 10% for validation (70–80%)
# last 20% (80–100%) for test

X_train, y_train = X.iloc[:train_end_idx],          y.iloc[:train_end_idx]
X_val,   y_val   = X.iloc[train_end_idx:val_end_idx], y.iloc[train_end_idx:val_end_idx]
X_test,  y_test  = X.iloc[val_end_idx:],           y.iloc[val_end_idx:]

print("Split sizes (by percentage indices):")
print(f"  Train: {X_train.shape[0]} rows ({train_end_idx} / {n})")
print(f"  Val  : {X_val.shape[0]} rows ({val_end_idx - train_end_idx} / {n})")
print(f"  Test : {X_test.shape[0]} rows ({n - val_end_idx} / {n})")
print()

# Optional: show date ranges for each split for your report
print("Train date range:", df['Date'].iloc[0].date(), "->", df['Date'].iloc[train_end_idx - 1].date())
print("Val date range  :", df['Date'].iloc[train_end_idx].date(), "->", df['Date'].iloc[val_end_idx - 1].date())
print("Test date range :", df['Date'].iloc[val_end_idx].date(), "->", df['Date'].iloc[-1].date())
print()


# -----------------------------
# 4. Baseline (always predict 'up')
# -----------------------------

# Useful for your report comparison
baseline_pred_val = np.ones_like(y_val)
baseline_pred_test = np.ones_like(y_test)

baseline_val_acc = accuracy_score(y_val, baseline_pred_val)
baseline_val_f1  = f1_score(y_val, baseline_pred_val)

baseline_test_acc = accuracy_score(y_test, baseline_pred_test)
baseline_test_f1  = f1_score(y_test, baseline_pred_test)

# -----------------------------
# 5. Models + evaluation
# -----------------------------

models = {
    "LogisticRegression": Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "KNN": Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=5  # you could tune this using the validation set
        ))
    ]),
    "DecisionTree": Pipeline(steps=[
        ("clf", DecisionTreeClassifier(
            max_depth=5,   # simple tree (you can also tune this)
            random_state=42
        ))
    ])
}


rows = []

for name, model in models.items():
    # Fit on train
    model.fit(X_train, y_train)

    # Validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1  = f1_score(y_val, y_val_pred)

    # Refit on train+val for test evaluation
    model.fit(
        pd.concat([X_train, X_val], axis=0),
        pd.concat([y_train, y_val], axis=0)
    )
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred)

    rows.append({
        "Model": name,
        "Val Accuracy":  val_acc,
        "Val F1":        val_f1,
        "Test Accuracy": test_acc,
        "Test F1":       test_f1
    })

# Add baseline row for comparison
rows.append({
    "Model": "Baseline (always up)",
    "Val Accuracy":  baseline_val_acc,
    "Val F1":        baseline_val_f1,
    "Test Accuracy": baseline_test_acc,
    "Test F1":       baseline_test_f1
})

results_df = pd.DataFrame(rows)

print("Model performance (validation and test):")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---- Figure 2: Test F1 comparison between models ----
plt.figure()
plt.bar(results_df['Model'], results_df['Test F1'])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Test F1 (Up class)")
plt.title("Comparison of models on test set")
plt.tight_layout()


# -----------------------------
# 6. Simple "helper" printout for the latest day
# -----------------------------

# For simplicity, we'll use Logistic Regression as the "best" model
best_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# Train on ALL available labeled data (train + val + test combined)
best_model.fit(X, y)

# Take the most recent row in the dataset
latest_row = df.iloc[-1]
latest_features = latest_row[feature_cols].to_frame().T


# Predict probability that SPY goes up tomorrow
proba_up = best_model.predict_proba(latest_features)[0, 1]
pred_label = best_model.predict(latest_features)[0]

print("\n--- Helper output for the most recent day in the dataset ---")
print(f"Date in data        : {latest_row['Date'].date()}")
print(f"SPY close on that day: {latest_row['Close']:.2f}")
print(f"Model P(UpTomorrow=1): {proba_up:.3f}")

if pred_label == 1:
    print("Model prediction     : SPY is more likely to CLOSE HIGHER the next trading day (UpTomorrow = 1).")
else:
    print("Model prediction     : SPY is more likely to CLOSE LOWER or about the same the next trading day (UpTomorrow = 0).")

# Show all open figures
plt.show()