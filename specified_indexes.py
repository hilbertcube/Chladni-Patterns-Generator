import matplotlib.pyplot as plt
import numpy as np
import itertools

# Settings here
title = True
font_title = 9.2        # Title font size
color_plots = True      # All black or color plots
thickness = 1.2         # Plot line thickness
start = 217             # Start at pattern #
end = 288               # End at pattern #

# Range and step size
delta = 0.005
xrange = np.arange(-1.0, 1.0, delta)
yrange = np.arange(-1.0, 1.0, delta)
x, y = np.meshgrid(xrange, yrange)

# Define the basis function
def X(x, n):
    return np.mod(n + 1, 2) * np.cos(n * np.pi * x / 2.0) + np.mod(n, 2) * np.sin(n * np.pi * x / 2.0)

# Sign combination generator
def generate_sign_combinations(pairs):
    sign_combinations = list(itertools.product([1, -1], repeat=(len(pairs) - 1)))
    return [[1] + list(comb) for comb in sign_combinations]  # Keep the first sign positive

# The wave function with contour cut z = 0
def u(x, y, pairs, signs):
    equation = np.zeros_like(x)  # Return an array of 0's with the same type
    for (n, m), sign in zip(pairs, signs):
        equation += sign * X(x, n) * X(y, m)
    return equation

# Searching combinations of sum of squares
def find_pairs(number):
    return [(n, m) for n in range(0, int(np.sqrt(number)) + 1)
            for m in range(0, int(np.sqrt(number)) + 1) if n**2 + m**2 == number]

# Check if a given number has sums of squares pairs
def exist_sos(number):
    return any(n**2 + m**2 == number for n in range(0, int(np.sqrt(number)) + 1)
               for m in range(0, int(np.sqrt(number)) + 1))

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'

# Plot the sum of squares patterns for specified N values
def plot_sum_of_squares(N_values):
    equations = []
    descriptions = []
    sums_of_squares = []

    for N in N_values:
        k = 0
        found_plots = 0
        while found_plots < N:
            if exist_sos(k):
                pairs = find_pairs(k)
                non_zero_pairs = pairs[1:-1]
                sign_combinations = generate_sign_combinations(pairs)
                for signs in sign_combinations:
                    found_plots += 1
                    if found_plots == N:
                        equations.append((pairs, signs, u(x, y, pairs, signs)))
                        descriptions.append(f"{pairs} {signs}")
                        sums_of_squares.append(k)
                        break
                if non_zero_pairs and found_plots < N:
                    sign_combinations_0 = generate_sign_combinations(non_zero_pairs)
                    for signs in sign_combinations_0:
                        found_plots += 1
                        if found_plots == N:
                            equations.append((non_zero_pairs, signs, u(x, y, non_zero_pairs, signs)))
                            descriptions.append(f"{non_zero_pairs} {signs}")
                            sums_of_squares.append(k)
                            break
            k += 1

    total_plots = len(equations)
    ncols = 7  # Fixed number of columns
    nrows = (total_plots + ncols - 1) // ncols  # Calculate the number of rows (// = Floor division)

    # Define fig and axis
    fig, axs = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, 3 * nrows))

    # Colors
    colors = ['darkblue', 'red', 'darkgreen', 'deeppink', 'darkviolet', 'blue', 'darkgoldenrod', 'teal', 'darkred', 'darkcyan']
    pair_color_mapping = {}
    color_index = 0

    # Plotting
    for i in range(total_plots):
        # Print all pairs and signs to the console
        print(f"N = {N_values[i]} | S = {sums_of_squares[i]}: {descriptions[i]}")

        # pairs = equations[i][0], signs = equations[i][1], equation = equations[i][2]
        pairs, signs, equation = equations[i]
        pairs_tuple = tuple(pairs)
        if pairs_tuple not in pair_color_mapping:
            pair_color_mapping[pairs_tuple] = colors[color_index % len(colors)]
            color_index += 1

        mode_color = pair_color_mapping[pairs_tuple] if color_plots else 'black'
        row, col = divmod(i, ncols)
        plot_title = rf"$N = {N_values[i]},\; S = {sums_of_squares[i]}$"

        ax = axs[row, col] if nrows > 1 else axs[col]
        ax.contour(x, y, equation, levels=[0], colors=mode_color, linewidths=thickness)

        if title:
            ax.set_title(plot_title, fontsize=font_title, ha='center', rotation='vertical', x=-0.1, y=0)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(hspace= 0.1, wspace= 0.3)
    plt.show()

# Plotting the sum of squares patterns for specific N values
N_values_to_plot = [21, 17, 33, 22, 62, 52, 233]  # Specify the N values to plot here
plot_sum_of_squares(N_values_to_plot)
