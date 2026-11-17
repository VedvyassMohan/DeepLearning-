import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

X, y = make_classification(n_samples=900, n_features=30, n_informative=6,
                           n_redundant=5, flip_y=0.12, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5, stratify=y)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

def build_model(use_dropout):
    layers = [Dense(128, activation='relu', input_shape=(X_train.shape[1],))]
    if use_dropout: layers.append(Dropout(0.5))
    layers.append(Dense(64, activation='relu'))
    if use_dropout: layers.append(Dropout(0.3))
    layers.append(Dense(1, activation='sigmoid'))
    model = Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

for label, use_dropout in [('Without dropout', False), ('With dropout', True)]:
    model = build_model(use_dropout)
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(label, 'Training accuracy:', round(float(train_acc), 4),
          'Testing accuracy:', round(float(test_acc), 4))
