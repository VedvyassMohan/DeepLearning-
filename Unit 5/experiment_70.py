from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=700, n_features=20, n_informative=8,
                           n_redundant=5, random_state=70)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=70, stratify=y)
original = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
reduced = make_pipeline(StandardScaler(), PCA(n_components=5),
                        LogisticRegression(max_iter=1000))
original.fit(Xtr, ytr); reduced.fit(Xtr, ytr)
a1 = accuracy_score(yte, original.predict(Xte))
a2 = accuracy_score(yte, reduced.predict(Xte))
print("Accuracy without PCA:", round(a1, 3))
print("Accuracy with PCA:", round(a2, 3))
print("Difference (PCA - original):", round(a2-a1, 3))
