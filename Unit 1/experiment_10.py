import numpy as np

rng = np.random.default_rng(42)
temperatures = rng.normal(loc=28, scale=3, size=1000)
print('First 10 observations:', np.round(temperatures[:10], 2))
print('Sample mean:', round(float(np.mean(temperatures)), 2))
print('Sample standard deviation:', round(float(np.std(temperatures, ddof=1)), 2))
print('Expected mean: 28; expected standard deviation: 3')
