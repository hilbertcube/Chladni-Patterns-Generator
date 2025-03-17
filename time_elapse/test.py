import numpy as np
import matplotlib.pyplot as plt

# Increase the number of points for t to make the curve smoother
t = np.logspace(1, 6, 10000)  # 10,000 points for better resolution

# Equation for the plot (use element-wise operations)
eq = (6 * np.cos(((10**5) * t) - (np.pi / 4))) / ((5 * np.cos((10**4) * t)) + (3 * np.cos((10**5) * t)))

# Plot the equation using semilogx for logarithmic scale on the x-axis
plt.figure()
plt.semilogx(t, eq)

# Set the title and axis labels
plt.title('G(jω) vs ω (rad/s)')
plt.xlabel('ω (rad/s)')
plt.ylabel('G(jω) (V)')

# Optionally, you can further refine the plot appearance:
plt.grid(True)  # Add a grid for better readability
plt.xticks(fontsize=12)  # Set font size for x-axis
plt.yticks(fontsize=12)  # Set font size for y-axis

# Show the plot
plt.show()
