import numpy as np

# Initialize weights
W = np.array([0.7, 0.5, 0.9])

# Initialize gradient
G = np.array([0.2, 0.1, 0.3])

# Set learning rate
learning_rate = 0.1

# Update weights using gradient descent
W_new = W - learning_rate * G

# Display updated weights
print("Original Weights:", W)
print("Updated Weights:", W_new)