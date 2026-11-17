import numpy as np

hours = np.array([2, 4, 6, 8, 10], dtype=float)
marks = np.array([35, 45, 60, 75, 85], dtype=float)
print('Sample covariance:', round(float(np.cov(hours, marks, ddof=1)[0, 1]), 2))
print('Correlation:', round(float(np.corrcoef(hours, marks)[0, 1]), 4))
