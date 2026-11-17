import numpy as np

rng = np.random.default_rng(78)
mean = np.array([25.0, 60.0, 1012.0])
cov = np.array([[4.0, 1.2, 0.5],
                [1.2, 16.0, -0.8],
                [0.5, -0.8, 9.0]])
samples = rng.multivariate_normal(mean, cov, size=1000)
estimated_mean = samples.mean(axis=0)
estimated_cov = np.cov(samples, rowvar=False)
print("Generated data shape:", samples.shape)
print("Expected mean:", mean)
print("Estimated mean:", np.round(estimated_mean, 2))
print("Expected covariance:\n", cov)
print("Estimated covariance:\n", np.round(estimated_cov, 2))
