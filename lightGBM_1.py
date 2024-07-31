import yfinance as yf
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Fetch historical data for a stock (e.g., Apple)
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Preprocessing
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Creating lagged features
for lag in range(1, 6):
    data[f'Lag_{lag}'] = data['Returns'].shift(lag)
data.dropna(inplace=True)

# Adding technical indicators
data['SMA_50'] = ta.sma(data['Close'], length=50)
data['RSI'] = ta.rsi(data['Close'], length=14)
data.dropna(inplace=True)

# Split data into features and target
X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'SMA_50', 'RSI']]
y = data['Returns']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train LightGBM model with early stopping callback
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Predictions
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
