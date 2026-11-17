import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(75)
n = 200
study = rng.normal(size=n)
attendance = 0.7*study + rng.normal(0, .5, n)
assignments = 0.8*study + rng.normal(0, .5, n)
participation = 0.5*study + rng.normal(0, .7, n)
marks = 0.75*study + rng.normal(0, .5, n)
X = np.c_[marks, attendance, assignments, participation]
X = StandardScaler().fit_transform(X)
fa = FactorAnalysis(n_components=2, random_state=75).fit(X)
scores = fa.transform(X)
print("Original variables: marks, attendance, assignments, participation")
print("Factor score shape:", scores.shape)
print("Component loading matrix:\n", np.round(fa.components_.T, 3))
