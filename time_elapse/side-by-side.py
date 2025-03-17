import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)

def g(x, y):
    return 3*(1 - x)**2 * np.exp(-x**2 - (y + 1)**2) - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) - 1/3*np.exp(-(x + 1)**2 - y**2)

x = np.linspace(-16, 16, 500)
y = np.linspace(-16, 16, 500)
X, Y = np.meshgrid(x, y)

x2 = np.linspace(-3, 3, 500)
y2 = np.linspace(-3, 3, 500)
X2, Y2 = np.meshgrid(x2, y2)

Z = f(X, Y)
Z2 = g(X2, Y2)

# Create a figure for side-by-side plots
fig = plt.figure(figsize=(16, 8))

# First plot
ax1 = fig.add_subplot(121, projection='3d')
surface1 = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
wireframe1 = ax1.plot_wireframe(X, Y, Z, color='k', linewidth=0.1)
ax1.set_box_aspect([1, 1, 0.4])
ax1.set_xlabel(rf"$x$", color='grey')
ax1.set_ylabel(rf"$y$", color='grey')
ax1.set_zlabel(rf"$z$", color='grey')
ax1.tick_params(axis='both', labelcolor='grey')
# ax1.set_axis_off()

# Second plot
ax2 = fig.add_subplot(122, projection='3d')
surface2 = ax2.plot_surface(X2, Y2, Z2, cmap='plasma', edgecolor='k')
wireframe2 = ax2.plot_wireframe(X2, Y2, Z2, color='k', linewidth=0.1)
ax2.set_box_aspect([1, 1, 0.4])
ax2.set_xlabel(rf"$x$", color='grey')
ax2.set_ylabel(rf"$y$", color='grey')
ax2.set_zlabel(rf"$z$", color='grey')
ax2.tick_params(axis='both', labelcolor='grey')
# ax2.set_axis_off()


plt.savefig('plot.png', transparent=True, dpi=300)
plt.show()