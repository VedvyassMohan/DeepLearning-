import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(74)
n = 1000
s1 = rng.uniform(-2, 2, n)
s2 = rng.laplace(0, 1, n)
S = np.c_[s1, s2]
A = np.array([[1, 2], [2, 1]])
X = StandardScaler().fit_transform(S @ A.T)
pca = PCA(n_components=2).fit_transform(X)
ica = FastICA(n_components=2, random_state=74, max_iter=1500).fit_transform(X)
print("PCA output shape:", pca.shape)
print("ICA output shape:", ica.shape)
print("PCA component correlation:\n", np.round(np.corrcoef(pca.T), 3))
print("ICA component correlation:\n", np.round(np.corrcoef(ica.T), 3))
print("PCA decorrelates components; ICA seeks statistically independent components.")
