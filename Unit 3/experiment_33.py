import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

X, y = make_classification(n_samples=600, n_features=12, n_informative=7,
                           class_sep=1.2, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLPClassifier provides SGD and Adam. RMSprop is implemented below
# with a small NumPy neural-style logistic model for comparison.
for solver in ['sgd', 'adam']:
    model = MLPClassifier(hidden_layer_sizes=(24,), solver=solver,
                          max_iter=300, random_state=10)
    model.fit(X_train, y_train)
    print(f'{solver.upper()}: iterations={model.n_iter_}, '
          f'loss={model.loss_:.4f}, accuracy={accuracy_score(y_test, model.predict(X_test)):.3f}')

# RMSprop logistic classifier trained by gradient descent
w = np.zeros(X_train.shape[1]); b = 0.0
avg_sq = np.zeros_like(w); beta = 0.9; lr = 0.01; eps = 1e-8
for epoch in range(300):
    z = X_train @ w + b
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    error = p - y_train
    gw = X_train.T @ error / len(y_train); gb = error.mean()
    avg_sq = beta * avg_sq + (1-beta) * (gw**2)
    w -= lr * gw / (np.sqrt(avg_sq) + eps); b -= lr * gb
p_test = 1 / (1 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
print(f'RMSPROP (logistic baseline): epochs=300, accuracy={accuracy_score(y_test, p_test >= 0.5):.3f}')
