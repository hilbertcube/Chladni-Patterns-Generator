import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def f(x, y, t):
    return 0.2 * np.cos(t) * (
        np.cos(2 * np.pi * x / 2) * np.cos(4 * np.pi * y / 2) +
        np.cos(4 * np.pi * x / 2) * np.cos(2 * np.pi * y / 2)
    )

# mesh
x = np.linspace(-1, 1, 40)
y = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


def update_surface(t):
    ax.clear()
    
    Z = f(X, Y, t)
    
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, rstride=1, cstride=1, linewidth=0)
    
    # Plot the central plane z = 0 as a grid
    ax.plot_wireframe(X, Y, np.zeros_like(X), color='black', alpha=0.5, linewidth=0.5)  
    
    # Plot the contour lines where f(x,y,t) = 0 on the plane
    ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2, linestyles='solid', zdir='z', offset=0)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 0.5)
    
    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    # Set equal scaling for axes
    ax.set_box_aspect([1, 1, 0.5])  # x:y:z
    return surface

# animation one full cycle (0 to 2π)
t_values = np.linspace(0, 2 * np.pi, 100)  # 100 frames for one cycle
ani_3d = animation.FuncAnimation(fig, update_surface, frames=t_values, interval=1, blit=False)

plt.show()

output_path = 'animation.gif'
writer = animation.PillowWriter(fps=50)
print("Saving...")
ani_3d.save(output_path, writer=writer)

print(f"Animation saved to: {output_path}")
