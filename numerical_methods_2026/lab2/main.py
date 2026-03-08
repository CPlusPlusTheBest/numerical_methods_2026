from csv_reader import read_data
import matplotlib.pyplot as plt

# Divided Differences for Newton Polynomial
# Formula: f[x0...xk] = (f[x1...xk] - f[x0...xk-1]) / (xk - x0)
def divided_differences(x, y):
    n = len(y)
    # Create a 2D table filled with zeros
    coef = [[0.0] * n for _ in range(n)]
    # The first column is the y values themselves
    for i in range(n):
        coef[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            # Applying the recursive formula
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    # Return the first row: f[x0], f[x0,x1], f[x0,x1,x2], etc.
    return [coef[0][i] for i in range(n)]


# Newton Interpolation Calculation
# Formula: P(x) = f[x0] + f[x0,x1]*(x-x0) + f[x0,x1,x2]*(x-x0)*(x-x1) + ...
def newton_interpolation(x_nodes, y_nodes, x_dest):
    coef = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    res = coef[0]
    product = 1.0
    for i in range(1, n):
        # Calculating the term (x - x0)*(x - x1)*...*(x - xi-1)
        product *= (x_dest - x_nodes[i - 1])
        res += coef[i] * product
    return res


# Finite Differences for Factorial Polynomials
# Formula: Delta^k f(x0) = Delta^{k-1} f(x1) - Delta^{k-1} f(x0)
def forward_differences(y):
    n = len(y)
    # Create a list to store the first entries of each difference level
    diffs = [y[0]]
    current_layer = y
    for i in range(1, n):
        # Calculate the next layer of differences
        next_layer = []
        for j in range(len(current_layer) - 1):
            next_layer.append(current_layer[j + 1] - current_layer[j])
        diffs.append(next_layer[0])
        current_layer = next_layer
    return diffs


# Interpolation using Factorial Polynomials (Newton's Forward Formula)
def factorial_interpolation(x_nodes, y_nodes, x_dest):
    h = x_nodes[1] - x_nodes[0]
    t = (x_dest - x_nodes[0]) / h
    diffs = forward_differences(y_nodes)

    res = diffs[0]
    t_factorial_product = 1.0
    factorial = 1.0

    for k in range(1, len(diffs)):
        # Update falling factorial: t^(k)
        t_factorial_product *= (t - (k - 1))
        # Update k!
        factorial *= k
        res += (diffs[k] * t_factorial_product) / factorial
    return res

# Plotting the interpolation results
def plot_interpolation(x_nodes, y_nodes, target_x, target_y):
    # num_points: number of points for a smooth curve
    num_points = 500
    x_min, x_max = min(x_nodes), max(x_nodes)

    # Generate x_range without numpy
    # Formula: x_i = x_min + i * (x_max - x_min) / (num_points - 1)
    x_range = [x_min + i * (x_max - x_min) / (num_points - 1) for i in range(num_points)]

    # Calculate polynomial values for each point in x_range
    y_plot = [newton_interpolation(x_nodes, y_nodes, xi) for xi in x_range]

    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.plot(x_nodes, y_nodes, 'ro', label='Experimental Data')

    # Plot the interpolation curve
    plt.plot(x_range, y_plot, 'b-', label='Newton Polynomial')

    # Highlight the predicted target point
    plt.plot(target_x, target_y, 'go', label=f'Prediction for {target_x}')

    # Chart styling
    plt.title('Training Time Interpolation')
    plt.xlabel('Dataset Size (n)')
    plt.ylabel('Time (sec)')
    plt.legend()
    plt.grid(True)
    plt.show()

def runge_func(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)

# Runge's Function: f(x) = 1 / (1 + 25 * x^2)
def runge_demo():

    plt.figure(figsize=(12, 8))
    # Generate plot points manually
    plot_points = 1000
    x_plot = [-1.0 + i * 2.0 / (plot_points - 1) for i in range(plot_points)]
    y_true = [runge_func(xi) for xi in x_plot]

    plt.plot(x_plot, y_true, 'k--', label='Original Runge Function')

    for n in [5, 10, 20]:
        # Generate n uniform nodes
        x_nodes = [-1.0 + i * 2.0 / (n - 1) for i in range(n)]
        y_nodes = [runge_func(xi) for xi in x_nodes]
        y_interp = [newton_interpolation(x_nodes, y_nodes, xi) for xi in x_plot]
        plt.plot(x_plot, y_interp, label=f'Nodes n={n}')

    plt.ylim(-1, 1.5)
    plt.title("Runge's Phenomenon: Oscillations at Interval Edges")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    try:
        x_data, y_data = read_data("data.csv")
        target_x = 120000.0

        # Prediction for 120,000 using Newton's method
        time_newton = newton_interpolation(x_data, y_data, target_x)

        # Prediction using Factorial method (using first 3 nodes with uniform step 10000)
        time_fact = factorial_interpolation(x_data[:3], y_data[:3], target_x)

        print(f"Time prediction for n={target_x}:")
        print(f"Newton Method: {time_newton:.2f} sec")
        print(f"Factorial Method (local uniform grid): {time_fact:.2f} sec")

        plot_interpolation(x_data, y_data, target_x, time_newton)
        runge_demo()
    except FileNotFoundError:
        print("Error: 'data.csv' not found. Please create it with ds,sec columns.")

