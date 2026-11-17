import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(76)
n = 250
price_focus = rng.normal(size=n)
quality_focus = rng.normal(size=n)
data = np.c_[0.8*price_focus+rng.normal(0,.4,n),
             0.7*price_focus+rng.normal(0,.5,n),
             0.8*quality_focus+rng.normal(0,.4,n),
             0.6*quality_focus+rng.normal(0,.6,n),
             0.3*price_focus+0.4*quality_focus+rng.normal(0,.6,n)]
X = StandardScaler().fit_transform(data)
fa = FactorAnalysis(n_components=2, random_state=76).fit(X)
print("Factor loadings (rows=behaviour variables, columns=factors):")
print(np.round(fa.components_.T, 3))
print("Latent factors summarize shared purchasing behaviour patterns.")
