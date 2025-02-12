import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def construct_design_matrix(X, degree):
    """
    Constructs a design matrix with polynomials up to the specified degree.
    """
    n = X.shape[0]
    columns = [np.ones(n)]
    for d in range(1, degree + 1):
        columns.append(X**d)
    return np.column_stack(columns)

def information_matrix(X, weights=None, epsilon=1e-6):
    """Calculates the weighted, regularized information matrix from the design matrix."""
    if weights is None:
        weights = np.ones(X.shape[0])
    M = X.T @ np.diag(weights) @ X
    return M + epsilon * np.eye(M.shape[0])

def variance_function(X, M_inv, new_x):
    """Calculates the variance function for a new point."""
    f_new = np.array([new_x**i for i in range(M_inv.shape[0])])
    return f_new.T @ M_inv @ f_new

def find_optimal_point(X, M_inv, x_values):
    """Evaluates the variance function across a range of x values to find the optimal point."""
    variances = [variance_function(X, M_inv, x) for x in x_values]
    max_index = np.argmax(variances)
    return x_values[max_index], variances[max_index]

def plot_convergence(x_values, variances, iteration, ax):
    """Updates the convergence plot with new variance data for each iteration."""
    # Define a list of distinct colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'navy']
    # Ensure there are enough colors, repeat the list if necessary
    if iteration >= len(colors):
        colors = colors * ((iteration // len(colors)) + 1)
    ax.plot(x_values, variances, label=f'Iteration {iteration + 1}', color=colors[iteration])
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Variance')
    ax.set_title('Convergence of D-optimal Design Algorithm')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

def d_optimal_design(x_range, degree, iterations=10, tolerance=0.01):
    """Implements Fedorov's D-optimal experimental design algorithm with visualization."""
    X = np.array([x_range[0], np.mean(x_range), x_range[-1]])[:, None]
    X = construct_design_matrix(X.flatten(), degree)
    weights = np.array([1/3, 1/3, 1/3])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(iterations):
        M = information_matrix(X, weights)
        M_inv = inv(M)
        x_values = np.linspace(min(x_range), max(x_range), 300)
        variances = [variance_function(X, M_inv, x) for x in x_values]

        x_new, max_variance = find_optimal_point(X, M_inv, x_values)
        if abs(max_variance - degree) <= tolerance:
            print("Convergence reached with max variance sufficiently close to p.")
            break

        alpha = (max_variance - degree) / (degree * (max_variance - 1))
        X = np.vstack([X, construct_design_matrix(np.array([x_new]), degree)])
        weights = np.append(weights * (1 - alpha), alpha)

        plot_convergence(x_values, variances, i, ax)
    
    plt.show()
    return X, weights
