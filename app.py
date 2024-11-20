import matplotlib.pyplot as plt  # noqa: D100
import numpy as np

# Example: Simulated position data
time = np.arange(0, 50, 0.5)  # Time points (0 to 50 seconds)
x = np.sin(time) + np.random.normal(0, 0.1, len(time))  # X-coordinate  # noqa: NPY002
y = np.cos(time) + np.random.normal(0, 0.1, len(time))  # Y-coordinate  # noqa: NPY002

# Calculate speed
speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / 0.5

# Plot trajectory
plt.plot(x, y, label="Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Rat's Movement in Cage")
plt.legend()
plt.show()
