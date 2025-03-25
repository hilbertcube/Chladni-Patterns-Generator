import numpy as np
import matplotlib.pyplot as plt

t = np.logspace(1, 6, 10000) # 10000 = resolution

# Define eq
eq = (6 * np.cos(((10**5) * t) - (np.pi / 4))) / ((5 * np.cos((10**4) * t)) + (3 * np.cos((10**5) * t)))

# Plot the equation using long scale
plt.figure()
plt.semilogx(t, eq)

# Set the title and axis labels
plt.title('G(jω) vs ω (rad/s)')
plt.xlabel('ω (rad/s)')
plt.ylabel('G(jω) (V)')


plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
