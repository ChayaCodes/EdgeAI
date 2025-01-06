import numpy as np
import matplotlib.pyplot as plt
from ex_data import finding_distance_true_x
from scipy import optimize

# Generate data
true_x = finding_distance_true_x
std_dev = 0.2  # Standard deviation
num_points = 100  # Number of points to generate
points = np.random.normal(loc=true_x, scale=std_dev, size=num_points)


def loss(guess, measurements):
    err = []

    for m in measurements:
        err.append((guess - m) ** 2)

    err = np.array(err)
    return np.mean(err)


# Plot
xs = np.linspace(-1, 5, 5)  # 200)
losses = [loss(val, points) for val in xs]
plt.plot(xs, losses, label="Loss function")
plt.scatter(points, [0] * len(points), c="blue", marker='|', s=100, label="Measurements")
plt.scatter(xs, [loss(h, points) for h in xs], c="red", s=100, label="Loss")
plt.legend()
plt.show()


###########################################

def grad(guess, measurements):
    return 2 * np.mean(guess - measurements)


# Gradient descent
alpha = 0.1
init_guess = 0.0
history = [init_guess]

x_guess = init_guess
for _ in range(50):
    x_guess -= alpha * grad(x_guess, points)
    history.append(x_guess)


# Plot
xs = np.linspace(-1, 5, 200)
losses = [loss(val, points) for val in xs]
plt.plot(xs, losses, label="Loss function")
plt.scatter(history, [loss(h, points) for h in history], c="red", label=f"GD steps till {x_guess:.2f}")
plt.scatter(points, [0] * len(points), c="blue", marker='|', s=100, label="Measurements")
plt.legend()
plt.show()

#####################################################

# result = optimize.minimize(loss, init_guess, args=(points,))
# x_opt = result.x[0]
# loss_opt = result.fun
# plt.plot(xs, losses, label="Loss function")
# plt.scatter(history, [loss(h, points) for h in history], c="red", label=f"GD steps till {x_guess:.2f}")
# plt.scatter(points, [0] * len(points), c="blue", marker='|', s=100, label="Measurements")
# plt.scatter(x_opt, loss_opt, color='green', zorder=5, label=f"Optimal x = {x_opt:.2f}")
# plt.legend()
# plt.show()

