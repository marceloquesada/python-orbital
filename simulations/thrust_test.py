from propagators import cowell_propagators, analytical_propagators
from perturbations import orbit_control
from orbital_utils import constants
from orbital_elements import tle, state_vectors, elements
from utils import csv, visualization

import numpy as np


t_initial = 0
t_max_days = 0.1
t_step = 1

oes_0 = np.array([constants.radius_earth + 500, 0.001, 0.0002, 200, 200, 0])
oes_t = np.array([constants.radius_earth + 500.1, 0.001, 0.0002, 200, 200, 0])
thrust_matrix = np.array([[1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0],
                         [1.0, 0.0, 1.0]])

t = np.arange(t_initial, t_max_days*(24*3600), t_step)
state_vector_0 = state_vectors.get_state_vectors(oes_0, constants.mu_earth)

# NUMERICAL PROPAGATOR
controller = orbit_control.Orbit_controller_SDRE(thrust_matrix, oes_t)
# drag_func = atmospheric_drag.Atmospheric_drag_perturbation_exp(drag_coef, gamma)
num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [controller])

t_num, X_num = num_prop.propagate(t, integration_method="RK45")

oes_num = num_prop.to_orbital_elements()

visualization.plot_classic_orbital_elements_overlay([t_num/(24*3600), oes_num])

# RELATIVE STATE (chaser - target) IN TARGET LVLH FRAME
import matplotlib.pyplot as plt

X_tgt = controller.X_t(t_num)
rho = np.zeros((3, len(t_num)))
rho_dot = np.zeros((3, len(t_num)))
for k in range(len(t_num)):
    R_lvlh, w = controller._lvlh(X_tgt[:, k])
    rho[:, k] = R_lvlh.T @ (X_num[0:3, k] - X_tgt[0:3, k])
    rho_dot[:, k] = R_lvlh.T @ (X_num[3:6, k] - X_tgt[3:6, k]) - np.cross([0.0, 0.0, w], rho[:, k])

rho_m = rho*1e3          # km -> m
rho_dot_m = rho_dot*1e3  # km/s -> m/s
t_min = t_num/60

labels = ["Radial", "Along-track", "Cross-track"]
colors = ["#2a78d6", "#eb6834", "#1baf7a"]

fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
panels = [
    (axs[0], rho_m, np.linalg.norm(rho_m, axis=0), "Relative position [m]"),
    (axs[1], rho_dot_m, np.linalg.norm(rho_dot_m, axis=0), "Relative velocity [m/s]"),
]
for ax, comps, norm, ylabel in panels:
    for comp, label, color in zip(comps, labels, colors):
        ax.plot(t_min, comp, color=color, linewidth=1.5, label=label)
    ax.plot(t_min, norm, color="#555555", linewidth=1.5, linestyle="--", label="Magnitude")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axs[0].set_title("Chaser relative to target (target LVLH frame)", loc="left")
axs[0].legend(loc="upper right", frameon=False, ncol=4)
axs[1].set_xlabel("Time [min]")

plt.tight_layout()
plt.show()
