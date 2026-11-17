import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr

rng = np.random.default_rng(73)
t = np.linspace(0, 10, 2000)
voice1 = np.sin(2*np.pi*2*t) + 0.3*np.sin(2*np.pi*5*t)
voice2 = np.sign(np.sin(2*np.pi*1.2*t)) * 0.7
S = np.c_[voice1, voice2]
mixing = np.array([[1.0, 0.6], [0.5, 1.0]])
X = S @ mixing.T + rng.normal(0, 0.01, S.shape)
estimated = FastICA(n_components=2, random_state=73, max_iter=1500).fit_transform(X)
# ICA output order and sign are arbitrary; report best absolute correlations.
for i in range(2):
    scores = [abs(pearsonr(S[:, j], estimated[:, i])[0]) for j in range(2)]
    print(f"Separated component {i+1}: best absolute correlation = {max(scores):.3f}")
print("Original and separated arrays:", S.shape, estimated.shape)
print("Higher absolute correlation indicates closer recovery of a source waveform.")
