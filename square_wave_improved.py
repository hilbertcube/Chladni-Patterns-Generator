import matplotlib.pyplot as plt
import numpy as np
import itertools
from functools import lru_cache

# Settings here
title = True
font_title = 9.2
color_plots = True
thickness = 1.5
start = 73
end = 144

# Range and step size
delta = 0.005
xrange = np.arange(-1.0, 1.0, delta)
yrange = np.arange(-1.0, 1.0, delta)
x, y = np.meshgrid(xrange, yrange)

# Precompute trig values for speed
@lru_cache(maxsize=None)
def X_precomputed(axis, n):
    """Returns X(axis, n) precomputed."""
    if axis == 'x':
        arr = x
    else:
        arr = y
    if n % 2 == 0:  # cos
        return np.cos(n * np.pi * arr / 2.0)
    else:  # sin
        return np.sin(n * np.pi * arr / 2.0)

def generate_sign_combinations(pairs):
    """Yield sign combinations without storing all in memory."""
    for comb in itertools.product([1, -1], repeat=len(pairs) - 1):
        yield (1,) + comb  # Keep first sign positive

# Vectorized wave function
def u_vec(pairs, signs):
    result = np.zeros_like(x)
    for (n, m), sign in zip(pairs, signs):
        result += sign * X_precomputed('x', n) * X_precomputed('y', m)
    return result

def u_vec_fully_vectorized(pairs, signs):
    """
    Fully vectorized version of u_vec.
    Computes the wave function over the meshgrid without any loops over (n, m) pairs.
    """
    if not pairs:
        return np.zeros_like(x)
    
    # Extract all n's and m's
    n_array = np.array([n for n, _ in pairs])
    m_array = np.array([m for _, m in pairs])
    signs_array = np.array(signs)[:, np.newaxis, np.newaxis]  # shape: (num_pairs, 1, 1)
    
    # Stack precomputed X arrays along first axis
    Xx_stack = np.stack([X_precomputed('x', n) for n in n_array], axis=0)  # shape: (num_pairs, grid_h, grid_w)
    Xy_stack = np.stack([X_precomputed('y', m) for m in m_array], axis=0)  # same shape
    
    # Compute all contributions at once using broadcasting
    result = np.sum(signs_array * Xx_stack * Xy_stack, axis=0)
    
    return result

@lru_cache(maxsize=None)
def find_pairs(number):
    root = int(np.sqrt(number))
    return [(n, m) for n in range(root + 1)
            for m in range(root + 1) if n**2 + m**2 == number]

@lru_cache(maxsize=None)
def exist_sos(number):
    return bool(find_pairs(number))

def plot_sum_of_squares(start_index, end_index):
    total_plots = end_index - start_index + 1
    if total_plots <= 0:
        print("Error: End_index must be greater than start_index.")
        return
    if total_plots > 100:
        print("Error: Cannot plot more than 100 plots at a time.")
        return

    equations, descriptions, sums_of_squares = [], [], []
    k = found_plots = 0

    while len(equations) < total_plots:
        if exist_sos(k):
            pairs = find_pairs(k)
            non_zero_pairs = pairs[1:-1]

            for signs in generate_sign_combinations(pairs):
                found_plots += 1
                if start_index <= found_plots <= end_index:
                    equations.append((pairs, signs, u_vec(pairs, signs)))
                    descriptions.append(f"{pairs} {signs}")
                    sums_of_squares.append(k)
                if found_plots >= end_index:
                    break

            if found_plots >= end_index:
                break

            if non_zero_pairs:
                for signs in generate_sign_combinations(non_zero_pairs):
                    found_plots += 1
                    if start_index <= found_plots <= end_index:
                        equations.append((non_zero_pairs, signs, u_vec(non_zero_pairs, signs)))
                        descriptions.append(f"{non_zero_pairs} {signs}")
                        sums_of_squares.append(k)
                    if found_plots >= end_index:
                        break

        k += 1

    ncols = 9
    nrows = (total_plots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, nrows))
    fig.patch.set_facecolor('black')

    for ax in axs.flat:
        ax.set_facecolor('black')
        ax.grid(True, color='gray')

    colors = ['darkblue', 'red', 'darkgreen', 'deeppink', 'darkviolet',
              'blue', 'darkgoldenrod', 'teal', 'darkred', 'darkcyan']
    pair_color_mapping = {}
    color_index = 0

    for i, (pairs, signs, equation) in enumerate(equations):
        print(f"N = {start + i} | S = {sums_of_squares[i]}: {descriptions[i]}")
        pairs_tuple = tuple(pairs)
        if pairs_tuple not in pair_color_mapping:
            pair_color_mapping[pairs_tuple] = colors[color_index % len(colors)]
            color_index += 1

        mode_color = pair_color_mapping[pairs_tuple] if color_plots else 'white'
        row, col = divmod(i, ncols)
        plot_title = rf"$N = {start + i},\; S = {sums_of_squares[i]}$"

        ax = axs[row, col] if nrows > 1 else axs[col]
        ax.contour(x, y, equation, levels=[0], colors=mode_color, linewidths=thickness)

        if title:
            ax.set_title(plot_title, fontsize=font_title,
                         ha='center', rotation='vertical', x=-0.1, y=0)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.show()

plot_sum_of_squares(start, end)
