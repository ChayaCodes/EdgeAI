import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_crater_data(num_points=50, center=(2, 3), radius=5, noise_std=0.2, random_state=42):
    """
    Generate random points around a circle (the crater boundary).
    :param num_points: Number of points to generate.
    :param center: Tuple (cx, cy) for the true center of the circle.
    :param radius: True radius of the circle.
    :param noise_std: Standard deviation of Gaussian noise added to the boundary.
    :param random_state: Seed for reproducibility.
    :return: Nx2 array of (x, y) points.
    """
    np.random.seed(random_state)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Ideal points on the circle
    x_ideal = center[0] + radius * np.cos(angles)
    y_ideal = center[1] + radius * np.sin(angles)
    # Add noise
    x_noisy = x_ideal + np.random.normal(scale=noise_std, size=num_points)
    y_noisy = y_ideal + np.random.normal(scale=noise_std, size=num_points)
    return np.column_stack((x_noisy, y_noisy))


def cost_function(params, points):
    """
    The cost function to minimize. We want to find (cx, cy, r)
    that best fits the circle to the given points.

    :param params: A list/array [cx, cy, r].
    :param points: Nx2 array of (x, y) points.
    :return: The sum of squared differences between the distance
             of each point to center and the radius.
    """
    cx, cy, r = params
    x = points[:, 0]
    y = points[:, 1]

    # Distance from each point to (cx, cy)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # We want these distances to be as close as possible to r
    # -> minimize sum of (distance - r)^2
    residuals = distances - r
    return np.sum(residuals ** 2)


def find_crater_center_radius(points, initial_guess=(0, 0, 1)):
    """
    Use scipy.optimize.minimize to find the best (cx, cy, r).

    :param points: Nx2 array of (x, y) points.
    :param initial_guess: Initial guess for [cx, cy, r].
    :return: The optimized values for [cx, cy, r].
    """
    result = minimize(cost_function, x0=initial_guess, args=(points,), method='BFGS')
    return result.x


if __name__ == "__main__":
    # 1. Generate input data (round crater boundary points)
    true_center = (2, 3)
    true_radius = 5
    data_points = generate_crater_data(num_points=50, center=true_center, radius=true_radius, noise_std=0.3)

    # 2. Visualize the input data
    plt.figure(figsize=(6, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], label='Crater boundary (data)', alpha=0.7)

    # 3. Solve the exercise with scipy.minimize
    initial_guess = (0, 0, 1)  # some naive initial guess
    cx_est, cy_est, r_est = find_crater_center_radius(data_points, initial_guess)

    # 4. Present the solution
    print(f"Estimated center = ({cx_est:.2f}, {cy_est:.2f})")
    print(f"Estimated radius = {r_est:.2f}")

    # Plot the found circle
    circle_angles = np.linspace(0, 2 * np.pi, 200)
    x_fit = cx_est + r_est * np.cos(circle_angles)
    y_fit = cy_est + r_est * np.sin(circle_angles)
    plt.plot(x_fit, y_fit, 'r-', label='Fitted circle')

    # Show the true circle for comparison
    x_true = true_center[0] + true_radius * np.cos(circle_angles)
    y_true = true_center[1] + true_radius * np.sin(circle_angles)
    plt.plot(x_true, y_true, 'g--', label='True circle')

    plt.axis('equal')
    plt.legend()
    plt.title("Crater Fitting with Scipy Minimize")
    plt.show()
