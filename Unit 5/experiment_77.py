import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error

X, _ = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
k = 3
pca = PCA(n_components=k).fit(X)
pca_reconstruction = pca.inverse_transform(pca.transform(X))
fa = FactorAnalysis(n_components=k, random_state=77).fit(X)
fa_scores = fa.transform(X)
fa_reconstruction = fa.mean_ + fa_scores @ fa.components_
print("Original dimensions:", X.shape[1])
print("Reduced dimensions:", k)
print("PCA reconstruction MSE:", round(mean_squared_error(X, pca_reconstruction), 4))
print("Factor Analysis reconstruction MSE:", round(mean_squared_error(X, fa_reconstruction), 4))
