import numpy as np

# Input features
X = np.array([4, 6, 8])

# Weights
W = np.array([0.5, 0.3, 0.2])

# Bias
b = 2

# Calculate dot product
dot_product = np.dot(X, W)

# Calculate predicted output
prediction = dot_product + b

# Display the result
print("Dot Product:", dot_product)
print("Predicted Output:", prediction)