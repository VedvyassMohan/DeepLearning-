import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=300, centers=3, n_features=2,
                  cluster_std=1.2, random_state=79)
X = StandardScaler().fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=79)
gmm.fit(X)
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)
print("Number of customer segments:", len(np.unique(labels)))
print("First 10 segment assignments:", labels[:10])
print("Membership probabilities for first 5 customers:\n",
      np.round(probabilities[:5], 3))
print("Average log-likelihood:", round(gmm.score(X), 3))
