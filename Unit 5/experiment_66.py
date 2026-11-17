import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense

rng = np.random.default_rng(66)
t = np.arange(900)
series = np.sin(t / 25) + 0.4 * np.sin(t / 5) + rng.normal(0, 0.03, len(t))
window = 60
X = np.array([series[i:i+window] for i in range(len(series)-window)])
y = series[window:]
split = 650
for name, layer in [("Simple RNN", SimpleRNN(32)), ("LSTM", LSTM(32))]:
    model = Sequential([Input((window, 1)), layer, Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X[:split, ..., None], y[:split], epochs=15, batch_size=32, verbose=0)
    pred = model.predict(X[split:, ..., None], verbose=0).ravel()
    print(name, "test MSE:", round(mean_squared_error(y[split:], pred), 5))
print("Lower test MSE indicates better prediction on this long-window task.")
