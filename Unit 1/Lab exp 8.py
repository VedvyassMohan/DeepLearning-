import numpy as np
import matplotlib.pyplot as plt

# Create the matrix
A = np.array([
    [4, 2],
    [2, 4]
])

# Calculate eigenvalues
eigenvalues = np.linalg.eigvals(A)

# Display eigenvalues
print("Eigenvalues:")
print(eigenvalues)

# Plot eigenvalues in a bar chart
plt.bar(["Eigenvalue 1", "Eigenvalue 2"], eigenvalues)
plt.xlabel("Eigenvalues")
plt.ylabel("Value")
plt.title("Eigenvalues of Matrix A")
plt.grid(axis="y")
plt.show()