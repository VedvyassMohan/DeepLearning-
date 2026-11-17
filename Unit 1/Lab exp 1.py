import numpy as np

# Create the image matrix
A = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Flatten the matrix
vector = A.flatten()

# Calculate the sum of pixel intensities
total = np.sum(A)

# Display the results
print("Original Image Matrix:")
print(A)

print("Flattened Vector:")
print(vector)

print("Total Pixel Intensity:", total)