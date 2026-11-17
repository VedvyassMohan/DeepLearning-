import numpy as np

X = np.array([[2, 1], [1, 2], [3, 2], [0, 1]], dtype=float)
y = np.array([1, 1, 1, 0])
w = np.zeros(2); bias = 0.0; lr = 0.1
for _ in range(20):
    for xi, target in zip(X, y):
        pred = int(np.dot(xi, w) + bias >= 1.5)
        error = target - pred
        w += lr * error * xi
        bias += lr * error
preds = (X @ w + bias >= 1.5).astype(int)
print('Weights:', np.round(w, 3))
print('Predictions:', preds)
print('Actual:', y)
print('Accuracy:', round(float(np.mean(preds == y))*100, 2), '%')
