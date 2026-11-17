import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Demonstration only: synthetic patient-like features, not real medical advice.
X, y = make_classification(n_samples=1000, n_features=12, n_informative=7,
                           n_redundant=2, weights=[0.6, 0.4], random_state=80)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=80, stratify=y)
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
model = Sequential([Input((Xtr.shape[1],)), Dense(32, activation="relu"),
                    Dropout(0.25), Dense(16, activation="relu"),
                    Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, epochs=20, batch_size=32, verbose=0)
prob = model.predict(Xte, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
print("Accuracy:", round(accuracy_score(yte, pred), 3))
print("Precision:", round(precision_score(yte, pred), 3))
print("Recall:", round(recall_score(yte, pred), 3))
print("F1-score:", round(f1_score(yte, pred), 3))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print("Example predicted class for first test patient:", int(pred[0]))
