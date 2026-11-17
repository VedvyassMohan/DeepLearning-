import numpy as np

# Feature vectors of two movies
A = np.array([2, 4, 6])
B = np.array([1, 3, 5])

# Perform vector addition
combined = A + B

# Calculate dot product
similarity = np.dot(A, B)

# Display results
print("Combined Representation:", combined)
print("Similarity Score:", similarity)