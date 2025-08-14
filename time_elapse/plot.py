import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'

t = 1.569

def f(x, y):
    return 0.2 * np.cos(t) * (np.cos(2 * np.pi * x / 2) * np.cos(4 * np.pi * y /2) + np.cos(4 * np.pi * x/2) * np.cos(2 * np.pi * y/2))

# mesh
x = np.linspace(-1.0, 1.0, 500)
y = np.linspace(-1.0, 1.0, 500)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
wireframe = ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.1)

ax.set_box_aspect([1,1,0.4])

ax.set_xlabel(rf"$x$", color='grey')
ax.set_ylabel(rf"$y$", color='grey')
ax.set_zlabel(rf"$z$", color='grey')
ax.tick_params(axis='both', labelcolor='grey')
#ax.set_title(rf"$t = 1$")
#fig.colorbar(surface, shrink=0.5, aspect=5)

plt.savefig('plot.png', transparent=True, dpi=300)
plt.show()