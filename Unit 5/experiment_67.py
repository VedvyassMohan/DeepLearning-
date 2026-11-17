import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, _ = make_classification(n_samples=250, n_features=10, n_informative=5,
                           n_redundant=2, random_state=67)
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
reduced = pca.fit_transform(X)
print("Original data shape:", X.shape)
print("Reduced data shape:", reduced.shape)
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print("Total variance retained:", round(pca.explained_variance_ratio_.sum(), 3))
