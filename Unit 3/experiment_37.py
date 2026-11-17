import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout

X, y = make_classification(n_samples=800, n_features=24, n_informative=6,
                           flip_y=0.1, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

for label, reg in [('Without regularization', False), ('With L2 regularization', True)]:
    penalty = regularizers.l2(0.01) if reg else None
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=penalty),
        Dropout(0.25) if reg else Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(label, 'train accuracy:', round(float(train_acc), 4),
          'test accuracy:', round(float(test_acc), 4))
