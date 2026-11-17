import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target
X_pca = PCA(n_components=2).fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X_pca, y, test_size=0.25,
                                      random_state=68, stratify=y)
model = LogisticRegression(max_iter=500).fit(Xtr, ytr)
pred = model.predict(Xte)
print("Original features:", iris.data.shape[1])
print("PCA features:", X_pca.shape[1])
print("Classification accuracy:", round(accuracy_score(yte, pred), 3))
