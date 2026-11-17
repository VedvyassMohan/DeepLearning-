import numpy as np

# Input matrix
X = np.array([
    [30, 60],
    [35, 70]
])

# Weight matrix
W = np.array([
    [0.2, 0.5],
    [0.4, 0.3]
])

# Multiply the matrices
output = np.dot(X, W)

# Display the results
print("Input Matrix:")
print(X)

print("Weight Matrix:")
print(W)

print("Predicted Output Matrix:")
print(output)