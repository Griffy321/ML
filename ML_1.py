import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt

# Fetch historical data for a stock (e.g., Apple)
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Preprocessing
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Example: Creating lagged features
for lag in range(1, 6):
    data[f'Lag_{lag}'] = data['Returns'].shift(lag)
data.dropna(inplace=True)

# Example: Adding technical indicators
data['SMA_50'] = ta.sma(data['Close'], length=50)
data['RSI'] = ta.rsi(data['Close'], length=14)
data.dropna(inplace=True)

# Machine Learning Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split data into features and target
X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'SMA_50', 'RSI']]
y = data['Returns']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Custom PandasData feed to include SMA and RSI indicators
class PandasData(bt.feeds.PandasData):
    lines = ('sma', 'rsi',)
    params = (
        ('sma', -1),
        ('rsi', -1),
    )

# Backtrader Strategy
class MLStrategy(bt.Strategy):
    params = (
        ('lookback', 5),  # Lookback period for lagged features
        ('threshold', 0.001),  # Threshold for making buy/sell decisions
    )

    def __init__(self):
        self.model = model
        self.data_close = self.datas[0].close
        self.sma = self.datas[0].sma
        self.rsi = self.datas[0].rsi

    def next(self):
        if len(self) > self.params.lookback:
            # Construct the features vector
            features = [
                self.data_close[-1], self.data_close[-2], self.data_close[-3],
                self.data_close[-4], self.data_close[-5], self.sma[0], self.rsi[0]
            ]
            prediction = self.model.predict([features])[0]
            if prediction > self.params.threshold:
                self.buy()
            elif prediction < -self.params.threshold:
                self.sell()

# Convert the dataframe to backtrader data feed
datafeed = PandasData(dataname=data)

# Backtrader cerebro setup
cerebro = bt.Cerebro()
cerebro.adddata(datafeed)
cerebro.addstrategy(MLStrategy)
cerebro.run()
cerebro.plot()
