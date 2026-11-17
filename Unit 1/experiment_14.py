import numpy as np

weights = np.array([0.7, 0.5, 0.9])
gradients = np.array([0.2, 0.1, 0.3])
learning_rate = 0.1
updated = weights - learning_rate * gradients
print('Original weights:', weights)
print('Updated weights:', np.round(updated, 3))
