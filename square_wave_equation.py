import matplotlib.pyplot as plt
import numpy as np
import itertools

# Settings here
title = True
font_title = 9.2
color_plots = True      # All white or color plots
thickness = 1.5
start = 73              # Start at pattern #
end = 144               # End at pattern #

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

# Plot the sum of squares patterns from start_index to end_index (1-based indexing)
def plot_sum_of_squares(start_index, end_index):
    total_plots = end_index - start_index + 1
    
    if end_index <= start_index:
        print("Error: End_index must be greater than start_index.\n")
        return
    if total_plots > 100:
        print("Error: Cannot plot more than 100 plots at a time.\n")
        return

    equations = []
    descriptions = []
    sums_of_squares = []
    
    # Iterating over all possible combinations
    k = 0
    found_plots = 0
    while len(equations) < total_plots:
        if exist_sos(k):
            pairs = find_pairs(k)
            non_zero_pairs = pairs[1:-1]
            sign_combinations = generate_sign_combinations(pairs)
            for signs in sign_combinations:
                found_plots += 1
                if start_index <= found_plots <= end_index:
                    equations.append((pairs, signs, u(x, y, pairs, signs)))
                    descriptions.append(f"{pairs} {signs}")
                    sums_of_squares.append(k)
                if found_plots >= end_index:
                    break
            if non_zero_pairs:
                sign_combinations_0 = generate_sign_combinations(non_zero_pairs)
                for signs in sign_combinations_0:
                    found_plots += 1
                    if start_index <= found_plots <= end_index:
                        equations.append((non_zero_pairs, signs, u(x, y, non_zero_pairs, signs)))
                        descriptions.append(f"{non_zero_pairs} {signs}")
                        sums_of_squares.append(k)
                    if found_plots >= end_index:
                        break
            if found_plots >= end_index:
                break
        k += 1

    ncols = 9
    nrows = (total_plots + ncols - 1) // ncols
    
    # Define fig and axis
    fig, axs = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, nrows))
    
    # Color
    fig.patch.set_facecolor('black')  # figure background
    for ax in axs.flat:
        ax.set_facecolor('black')  # axes background
        ax.grid(True, color='gray')
    
    # Colors
    colors = ['darkblue', 'red', 'darkgreen', 'deeppink', 'darkviolet', 'blue', 'darkgoldenrod', 'teal', 'darkred', 'darkcyan']
    pair_color_mapping = {}
    color_index = 0
    
    # Plotting
    for i in range(total_plots):
        # Print all pairs and signs to the console
        print(f"N = {start + i} | S = {sums_of_squares[i]}: {descriptions[i]}")
        
        # pairs = equations[i][0], signs = equations[i][1], equation = equations[i][2]
        pairs, signs, equation = equations[i]
        pairs_tuple = tuple(pairs)
        if pairs_tuple not in pair_color_mapping:
            pair_color_mapping[pairs_tuple] = colors[color_index % len(colors)]
            color_index += 1
        
        if color_plots:
            mode_color = pair_color_mapping[pairs_tuple] 
        else :
            'white'
            
        row, col = divmod(i, ncols)
        plot_title = rf"$N = {start + i},\; S = {sums_of_squares[i]}$"
        
        ax = axs[row, col] if nrows > 1 else axs[col]
        ax.contour(x, y, equation, levels=[0], colors=mode_color, linewidths=thickness)
        
        if title:
            ax.set_title(plot_title, fontsize=font_title, ha='center', rotation='vertical', x=-0.1, y=0)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.show()

# Plotting the sum of squares patterns for all perfect squares
plot_sum_of_squares(start, end)
