import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense

X, y = make_classification(n_samples=700, n_features=20, n_informative=7, random_state=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
for label, penalty in [('No L2', 0.0), ('With L2', 0.01)]:
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],),
              kernel_regularizer=regularizers.l2(penalty)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(penalty)),
        Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    weight_norm = np.sqrt(sum(np.sum(w*w) for w in model.get_weights() if w.ndim > 1))
    print(label, 'weight norm:', round(float(weight_norm), 3),
          'test accuracy:', round(float(model.evaluate(X_test, y_test, verbose=0)[1]), 3))
