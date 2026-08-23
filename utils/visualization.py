import matplotlib.pyplot as plt
import numpy as np
from orbital_utils.constants import *
from cycler import cycler



def plot_classic_orbital_elements(t: np.typing.NDArray, orbital_elementss: np.typing.NDArray):
    """
    Plots the classic orbital elements over time.

    Parameters:
    t (np.ndarray): Time array.
    orbital_elements (list): List of orbital elements objects.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs[0, 0].plot(t, orbital_elementss[0, :], label='Major Axis')
    axs[0, 0].set_title('Major Axis')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Major Axis (km)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 1].plot(t, orbital_elementss[1, :], label='Eccentricity', color='orange')
    axs[0, 1].set_title('Eccentricity')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Eccentricity')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[1, 0].plot(t, orbital_elementss[2, :], label='Inclination', color='green')
    axs[1, 0].set_title('Inclination')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Inclination (degrees)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 1].plot(t, orbital_elementss[3, :], label='Ascending Node', color='red')
    axs[1, 1].set_title('Ascending Node')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Ascending Node (degrees)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[2, 0].plot(t, orbital_elementss[4, :], label='Argument of Perigee', color='purple')
    axs[2, 0].set_title('Argument of Perigee')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Argument of Perigee (degrees)')
    axs[2, 0].grid(True)
    axs[2, 0].legend()
    axs[2, 1].plot(t, orbital_elementss[5, :], label='True Anomaly', color='brown')
    axs[2, 1].set_title('True Anomaly')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    plt.tight_layout()
    plt.show()


def plot_classic_orbital_elements_overlay(*orbital_elementss_lists: list[list[np.typing.NDArray]]):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for orbital_elementss_list in orbital_elementss_lists:
        t = orbital_elementss_list[0]
        orbital_elementss = orbital_elementss_list[1]

        """
        Plots the classic orbital elements over time.

        Parameters:
        t (np.ndarray): Time array.
        orbital_elements (list): List of orbital elements objects.
        """
        axs[0, 0].plot(t, orbital_elementss[0, :], label='Major Axis')
        axs[0, 0].set_title('Major Axis')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Major Axis (km)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        axs[0, 1].plot(t, orbital_elementss[1, :], label='Eccentricity')
        axs[0, 1].set_title('Eccentricity')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Eccentricity')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        axs[1, 0].plot(t, orbital_elementss[2, :], label='Inclination')
        axs[1, 0].set_title('Inclination')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Inclination (degrees)')
        axs[1, 0].set_ylim(0, 180)
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        axs[1, 1].plot(t, orbital_elementss[3, :], label='Ascending Node')
        axs[1, 1].set_title('Ascending Node')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Ascending Node (degrees)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        axs[2, 0].plot(t, orbital_elementss[4, :], label='Argument of Perigee')
        axs[2, 0].set_title('Argument of Perigee')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Argument of Perigee (degrees)')
        axs[2, 0].grid(True)
        axs[2, 0].legend()
        axs[2, 1].plot(t, orbital_elementss[5, :], label='True Anomaly')
        axs[2, 1].set_title('True Anomaly')
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    plt.tight_layout()
    plt.show()


def plot_3D_view(
        t,
        X,
        plot_earth: bool = True,
        earth_radius: float = 6378.0,
        earth_color: str = 'blue',
        earth_alpha: float = 0.3
        ):
    plt.figure()
    if plot_earth:
        ax = plt.axes(projection='3d')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = earth_radius * np.cos(u)*np.sin(v)
        y = earth_radius * np.sin(u)*np.sin(v)
        z = earth_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=earth_color, alpha=earth_alpha)

    i = Bases.i
    j = Bases.j
    k = Bases.k

    ax.plot3D(X[0], X[1], X[2], 'b-')
    ax.quiver(0, 0, 0, i[0], i[1], i[2], length = 1.5*earth_radius)
    ax.quiver(0, 0, 0, j[0], j[1], j[2], length = 1.5*earth_radius)
    ax.quiver(0, 0, 0, k[0], k[1], k[2], length = 1.5*earth_radius)
    ax.set_title('Orbit Propagation')
    ax.axis('equal')
    plt.show()

def plot_3D_overlay(
        *Xs,
        plot_earth: bool = True,
        earth_radius: float = 6378.0,
        earth_color: str = 'blue',
        earth_alpha: float = 0.3,
        orbit_marker: str = '-'
        ):
    plt.figure()
    plt.style.use('bmh')
    if plot_earth:
        ax = plt.axes(projection='3d')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = earth_radius * np.cos(u)*np.sin(v)
        y = earth_radius * np.sin(u)*np.sin(v)
        z = earth_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=earth_color, alpha=earth_alpha)

    for i in range(len(Xs)):
        X = Xs[i]
        ax.plot3D(X[0], X[1], X[2], linewidth=0.7)

    i = Bases.i
    j = Bases.j
    k = Bases.k

    ax.quiver(0, 0, 0, i[0], i[1], i[2], length = 1.5*earth_radius)
    ax.quiver(0, 0, 0, j[0], j[1], j[2], length = 1.5*earth_radius)
    ax.quiver(0, 0, 0, k[0], k[1], k[2], length = 1.5*earth_radius)
    ax.set_title('Orbit Propagation')
    ax.axis('equal')
    plt.show()

