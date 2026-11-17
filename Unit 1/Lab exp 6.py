import numpy as np
import matplotlib.pyplot as plt

# Store the data
hours = np.array([2, 4, 6, 8, 10])
marks = np.array([35, 45, 60, 75, 85])

# Calculate correlation coefficient
correlation = np.corrcoef(hours, marks)[0, 1]

# Display the correlation
print("Correlation Coefficient:", round(correlation, 2))

# Plot the data
plt.scatter(hours, marks)
plt.xlabel("Study Hours")
plt.ylabel("Examination Marks")
plt.title("Study Hours vs Examination Marks")
plt.grid(True)
plt.show()