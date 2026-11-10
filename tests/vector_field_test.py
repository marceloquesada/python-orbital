"""
Plot the 3D vector field defined by get_local_atmosphere_velocity_vector.
"""

import numpy as np
import matplotlib.pyplot as plt

# Angular speed used inside get_local_atmosphere_velocity_vector
w = 0.2


def get_local_atmosphere_velocity_vector(state_vector):
    r_vec = state_vector[0:2]

    v_vec = np.array([w*r_vec[1], -w*r_vec[0], 0])

    return v_vec


def build_field(extent=50, n_points=10):
    """Evaluate the velocity field on a regular 3D grid of points."""
    coords = np.linspace(-extent, extent, n_points)
    X, Y, Z = np.meshgrid(coords, coords, coords)

    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    W = np.zeros_like(Z, dtype=float)

    it = np.nditer(X, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        state_vector = np.array([X[idx], Y[idx], Z[idx]])
        vx, vy, vz = get_local_atmosphere_velocity_vector(state_vector)
        U[idx], V[idx], W[idx] = vx, vy, vz

    return X, Y, Z, U, V, W


def plot_field(X, Y, Z, U, V, W):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.quiver(X, Y, Z, U, V, W, color="teal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Local atmosphere velocity field")

    plt.tight_layout()
    plt.savefig("atmosphere_velocity_field.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X, Y, Z, U, V, W = build_field(extent=50, n_points=10)
    plot_field(X, Y, Z, U, V, W)