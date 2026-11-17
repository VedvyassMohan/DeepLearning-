import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Features: area (1000 sq.ft units), bedrooms, location score
X = np.array([[1.0, 2, 1], [1.2, 2, 2], [1.5, 3, 2], [1.8, 3, 3], [2.0, 4, 3], [2.2, 4, 4]])
y = np.array([30, 36, 45, 54, 60, 68])  # price in lakh
model = LinearRegression().fit(X, y)
pred = model.predict(X)
print('Predicted prices:', np.round(pred, 2))
print('MAE:', round(mean_absolute_error(y, pred), 3))
print('MSE:', round(mean_squared_error(y, pred), 3))
print('R2:', round(model.score(X, y), 4))
