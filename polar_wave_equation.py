import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.special import jn, jnp_zeros

a = 1.0  # Radius of the plate
thickness = 1
color = 'red'
title = True
columns = 8
rows = 6

# Function to get the m-th zero of the derivative of the Bessel function of order n
def bessel_derivative_zero(n, m):
    if n == 0: #Redefine the first zero of J_0 prime to be 0
        zeros = list(jnp_zeros(n, m + 1))
        zeros.insert(0, 0)
        return zeros[m]
    else:
        return jnp_zeros(n, m + 1)[-1]  # Get the m-th zero, treating first zero as the 0th zero


# Initial function f(r, theta)
def f(r, theta):
    return r * np.cos(theta)

# Define the double integrals for coefficients a_nm and b_nm
def a_nm(n, m, a):
    z_nm = bessel_derivative_zero(n, m)
    numerator = integrate.dblquad(lambda theta, r: jn(n, z_nm * r * 1.0/a) * np.cos(n * theta) * 
                        f(r, theta) * r, 0, a, lambda r: 0, lambda r: 2 * np.pi)[0]
    denominator = integrate.quad(lambda r: jn(n, z_nm * r * 1.0/a) ** 2 * r, 0, a)[0]
    return numerator / (2 * np.pi * denominator)

def b_nm(n, m, a):
    z_nm = bessel_derivative_zero(n, m)
    numerator = integrate.dblquad(lambda theta, r: jn(n, z_nm * r * 1.0/a) * np.sin(n * theta) * 
                        f(r, theta) * r, 0, a, lambda r: 0, lambda r: 2 * np.pi)[0]
    denominator = integrate.quad(lambda r: jn(n, z_nm * r * 1.0/a) ** 2 * r, 0, a)[0]
    return numerator / (2 * np.pi * denominator)

# Define the function J_n(z_nm * r) * (a_nm * cos(n*theta) + b_nm * sin(n*theta))
def u(n, m, r, theta, a):
    z_nm = bessel_derivative_zero(n, m)
    A_nm = a_nm(n, m, a)
    B_nm = b_nm(n, m, a)
    return jn(n, z_nm * r * 1.0/a) * (A_nm * np.cos(n * theta) + B_nm * np.sin(n * theta))

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'

plt.rcParams['figure.facecolor'] = 'black'  # figure background
plt.rcParams['axes.facecolor'] = 'black'  # axes background
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Plotting function in polar coordinates
def plot_polar_contour(n_max, m_max, a, resolution=100):
    r = np.linspace(0, a, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    R, Theta = np.meshgrid(r, theta)
    
    if n_max < 1 or m_max < 1:
        print("Columns or lines cannot be less than 1\n")
        return

    # Define fig and axis
    fig, axs = plt.subplots(m_max, n_max, figsize=(1.4* n_max, 1.2 * m_max), subplot_kw={'projection': 'polar'})
        
    for m in range(m_max):
        for n in range(n_max):
            Z = u(n, m, R, Theta, a)
            ax = axs[m, n] if m_max > 1 else axs[n]
            ax.contour(Theta, R, Z, levels=[0], colors=color, linewidths=thickness)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            
            plot_title = rf"$(n,m) = {n, m}$"
            if title:
                ax.set_title(plot_title, fontsize = 9, ha='center', rotation='vertical', x=-0.1, y=0)

    plt.tight_layout()
    plt.show()

plot_polar_contour(columns, rows, a)