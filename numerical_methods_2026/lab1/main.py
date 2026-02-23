import requests
import numpy as np
import matplotlib.pyplot as plt

def fetch_elevation_data(url):
    """Fetches altitude data from Open-Elevation API."""
    response = requests.get(url)
    response.raise_for_status()
    raw_data = response.json()["results"]
    elevations = np.array([p["elevation"] for p in raw_data])
    return raw_data, elevations


def calculate_distances(raw_data):
    """
    Calculates distance between GPS coordinates.
    1. Convert degrees to radians.
    2. Apply Haversine formula to get distance on a sphere.
    """
    R = 6371000
    dist_list = [0]
    for i in range(1, len(raw_data)):
        p1, p2 = raw_data[i - 1], raw_data[i]
        phi1, phi2 = np.radians(p1["latitude"]), np.radians(p2["latitude"])
        dphi = np.radians(p2["latitude"] - p1["latitude"])
        dlambda = np.radians(p2["longitude"] - p1["longitude"])

        # Spherical law of cosines component
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        d = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist_list.append(dist_list[-1] + d)
    return np.array(dist_list)


def solve_tridiagonal(A, B, C, F):
    """
    Thomas Algorithm (Method Progonki)
    System: A[i]*x[i-1] + B[i]*x[i] + C[i]*x[i+1] = F[i]

    Step 1: Forward sweep (Calculating alpha and beta)
    alpha[i] = -C[i] / (B[i] + A[i]*alpha[i-1])
    beta[i]  = (F[i] - A[i]*beta[i-1]) / (B[i] + A[i]*alpha[i-1])
    """
    n = len(F)
    alpha, beta = np.zeros(n), np.zeros(n)
    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]
    for i in range(1, n - 1):
        denom = B[i] + A[i - 1] * alpha[i - 1]
        alpha[i] = -C[i] / denom
        beta[i] = (F[i] - A[i - 1] * beta[i - 1]) / denom

    """
    Step 2: Backward sweep (Finding unknowns x)
    x[n] = (F[n] - A[n]*beta[n-1]) / (B[n] + A[n]*alpha[n-1])
    x[i] = alpha[i]*x[i+1] + beta[i]
    """
    x = np.zeros(n)
    x[n - 1] = (F[n - 1] - A[n - 2] * beta[n - 2]) / (B[n - 1] + A[n - 2] * alpha[n - 2])
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return x


def get_coefficients(x_nodes, y_nodes):
    """
    Spline Coefficient Calculation
    Step 1: h[i] = x[i+1] - x[i]
    Step 2: a[i] = y[i]
    """
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    a = np.array(y_nodes[:-1])

    """
    Step 3: Setup tridiagonal system for 'c' coefficients:
    h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = F[i]
    where F[i] = 3 * ( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
    """
    size = n + 1
    A_low, B_diag, C_up, F_vec = np.zeros(size - 1), np.ones(size), np.zeros(size - 1), np.zeros(size)

    for i in range(1, n):
        A_low[i - 1] = h[i - 1]
        B_diag[i] = 2 * (h[i - 1] + h[i])
        C_up[i] = h[i]

        # Divided differences for the vector F
        slope_right = (y_nodes[i + 1] - y_nodes[i]) / h[i]
        slope_left = (y_nodes[i] - y_nodes[i - 1]) / h[i - 1]
        F_vec[i] = 3 * (slope_right - slope_left)

    # Solve for c using Thomas algorithm
    c_full = solve_tridiagonal(A_low, B_diag, C_up, F_vec)

    """
    Step 4: Calculate b[i] and d[i]:
    d[i] = (c[i+1] - c[i]) / (3 * h[i])
    b[i] = (y[i+1] - y[i])/h[i] - (h[i] * (c[i+1] + 2*c[i])) / 3
    """
    b, d = np.zeros(n), np.zeros(n)
    for i in range(n):
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (c_full[i + 1] + 2 * c_full[i]) / 3
        d[i] = (c_full[i + 1] - c_full[i]) / (3 * h[i])

    return a, b, c_full, d


def calculate_route_stats(distances, elevations):
    """
    Step 1: Total Length (last element of distance array)
    Step 2: Total Ascent (sum of positive elevation differences)
    Step 3: Total Descent (sum of negative elevation differences)
    """
    total_dist = distances[-1]
    n = len(elevations)
    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))

    print(f"Total Distance: {total_dist:.2f} m")
    print(f"Total Ascent:   {total_ascent:.2f} m")
    print(f"Total Descent:  {total_descent:.2f} m")
    return total_ascent


def calculate_energy_work(total_ascent):
    """Mechanical work: W = m * g * h_ascent (m=80kg, g=9.81)"""
    mass = 80
    g = 9.81
    energy_j = mass * g * total_ascent
    print(f"Mechanical Work: {energy_j:.2f} J ({energy_j / 1000:.2f} kJ)")
    print(f"Energy Burnt:    {energy_j / 4184:.2f} kcal")


def analyze_gradients(distances, elevations):
    """Grade (%) = (dy / dx) * 100"""
    a, b, c, d = get_coefficients(distances, elevations)
    gradients = [b[i] * 100 for i in range(len(distances) - 1)]

    print(f"Max Incline:      {np.max(gradients):.2f} %")
    print(f"Max Decline:      {np.min(gradients):.2f} %")
    print(f"Average Gradient: {np.mean(np.abs(gradients)):.2f} %")


def print_coeff_table(a, b, c, d, title):
    print(f"\n{title}:")
    print(f"{'Segment':<8} | {'a_i':^9} | {'b_i':^9} | {'c_i':^9} | {'d_i':^9}")
    print("-" * 65)
    for i in range(len(a)):
        print(f" S{i:2d}     | {a[i]:9.2f} | {b[i]:9.4f} | {c[i]:9.6f} | {d[i]:9.8f}")
    print(f"Last Node| {'-':^9} | {'-':^9} | {c[-1]:9.6f} | {'-':^9}")


if __name__ == "__main__":
    url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

    # 1. Get Data
    raw_data, elevations = fetch_elevation_data(url)
    distances = calculate_distances(raw_data)

    # 2. Stage 1: Full Analysis
    print("STAGE 1: ORIGINAL FULL DATA")
    a_f, b_f, c_f, d_f = get_coefficients(distances, elevations)
    print_coeff_table(a_f, b_f, c_f, d_f, "Coefficients (n=21)")

    print("\nROUTE CHARACTERISTICS")
    total_ascent = calculate_route_stats(distances, elevations)

    print("\nENERGY ANALYSIS")
    calculate_energy_work(total_ascent)

    print("\nGRADIENT ANALYSIS")
    analyze_gradients(distances, elevations)

    # --- 3. Stage 2: Comparative Visualization & Tables ---
    print("\nSTAGE 2: COMPARATIVE ANALYSIS (10, 15, 20 NODES)")
    node_counts = [10, 15, 20]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    for i, count in enumerate(node_counts):
        # Select a subset of nodes from the original data
        idx = np.linspace(0, len(distances) - 1, count, dtype=int)
        x_nodes, y_nodes = distances[idx], elevations[idx]

        # Calculate coefficients for this specific subset
        a, b, c, d = get_coefficients(x_nodes, y_nodes)

        # Print numerical results in the console
        print_coeff_table(a, b, c, d, f"Coefficients for {count} Nodes")

        # Step-by-step plotting for each spline segment
        for j in range(len(x_nodes) - 1):
            # Create 50 points between two nodes for a smooth curve
            xf = np.linspace(x_nodes[j], x_nodes[j + 1], 50)
            dx = xf - x_nodes[j]

            # S_j(x) = a_j + b_j(dx) + c_j(dx)^2 + d_j(dx)^3
            yf = a[j] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3
            axs[i].plot(xf, yf, 'b-', linewidth=1.5)

        # Overlay original data points (red) and selected nodes (black X)
        axs[i].scatter(distances, elevations, c='red', s=10, alpha=0.3, label='Original Data')
        axs[i].scatter(x_nodes, y_nodes, c='black', marker='x', s=40, label='Selected Nodes')
        axs[i].set_title(f"Spline Interpolation with {count} Nodes")
        axs[i].set_ylabel("Elevation (m)")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)

    plt.xlabel("Distance (m)")
    plt.tight_layout()
    plt.show()