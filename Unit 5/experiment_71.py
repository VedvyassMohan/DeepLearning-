import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, _ = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
pca = PCA().fit(X)
cumulative = np.cumsum(pca.explained_variance_ratio_)
required = np.argmax(cumulative >= 0.95) + 1
print("Total original features:", X.shape[1])
print("Components required for at least 95% variance:", required)
print("Variance retained:", round(cumulative[required-1], 4))
print("Cumulative explained variance:", np.round(cumulative, 3))
