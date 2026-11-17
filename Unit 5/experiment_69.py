import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, labels = make_blobs(n_samples=300, centers=4, n_features=6,
                       cluster_std=2.0, random_state=69)
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
points = pca.fit_transform(X)
plt.figure(figsize=(8, 5))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis", s=25)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Dataset Visualized using PCA")
plt.tight_layout()
plt.savefig("experiment_69_pca_plot.png", dpi=150)
plt.show()
print("Reduced shape:", points.shape)
print("Variance retained:", round(pca.explained_variance_ratio_.sum(), 3))
print("Plot saved as experiment_69_pca_plot.png")
