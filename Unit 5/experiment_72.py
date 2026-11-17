import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

rng = np.random.default_rng(72)
t = np.linspace(0, 8, 1000)
s1 = np.sin(2 * np.pi * 2 * t)
s2 = np.sign(np.sin(2 * np.pi * 3 * t))
sources = np.c_[s1, s2]
A = np.array([[1, 0.5], [0.4, 1.0]])
mixed = sources @ A.T + 0.02 * rng.normal(size=sources.shape)
ica = FastICA(n_components=2, random_state=72, max_iter=1000, tol=0.001)
separated = ica.fit_transform(mixed)
fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, sources); ax[0].set_title("Original Sources")
ax[1].plot(t, mixed); ax[1].set_title("Mixed Signals")
ax[2].plot(t, separated); ax[2].set_title("ICA Separated Components")
plt.tight_layout(); plt.savefig("experiment_72_ica.png", dpi=150); plt.show()
print("Mixed signal shape:", mixed.shape)
print("Separated signal shape:", separated.shape)
print("Plot saved as experiment_72_ica.png")
