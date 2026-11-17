import numpy as np
import matplotlib.pyplot as plt

# Store temperature and humidity values
temperature = np.array([22, 25, 28, 30, 32])
humidity = np.array([80, 75, 68, 60, 55])

# Combine the data
data = np.vstack((temperature, humidity))

# Calculate covariance matrix
covariance = np.cov(data)

# Display the covariance matrix
print("Covariance Matrix:")
print(covariance)

# Visualize the relationship
plt.scatter(temperature, humidity)
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("Temperature vs Humidity")
plt.grid(True)
plt.show()