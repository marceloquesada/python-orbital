from propagators import cowell_propagators
from perturbations import orbit_control
from orbital_utils import constants, ref_systems
from orbital_elements import state_vectors

import numpy as np
import matplotlib
matplotlib.use("Agg")  # file output only, no GUI window needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# SIMULATION SETUP
t_initial = 0
t_max_minutes = 15
t_step = 1

# SATELLITE SETUP
oes_0 = np.array([constants.radius_earth + 500, 0.001, 0.0002, 200, 200, 0])
mass = 2 # [Kg]

# CONTROLLER SETUP
oes_t = np.array([constants.radius_earth + 500.5, 0.001, 0.0002, 200, 200, 0])

# THRUSTER SETUP
thrust_matrix = np.array([[1.0, 0.0, 0.0],
                          [1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0]])
thrust_min = 1e-4 # [N]
thrust_max = 0.5  # [N]
Isp = 580         # [Kg]
mass_dry = 2.2    # [Kg]
mass_prop = 0.15  # [Kg]


t = np.arange(t_initial, t_max_minutes*60, t_step)
state_vector_0 = state_vectors.get_state_vectors(oes_0, constants.mu_earth)

controller = orbit_control.Orbit_controller_SDRE(thrust_matrix, oes_t, thrust_min/mass, thrust_max/mass)
num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [controller])
t_num, X_sat = num_prop.propagate(t, integration_method="RK45")
X_tgt = controller.X_t(t_num)

# RELATIVE STATE IN TARGET LVLH FRAME [m]
rho = np.zeros((3, len(t_num)))
for k in range(len(t_num)):
    rho[:, k] = ref_systems.inertial_to_lvlh(X_sat[0:3, k] - X_tgt[0:3, k], X_tgt[:, k], constants.mu_earth) * 1e3
radial, along, cross = rho

# ANIMATION
n_frames = 150
fps = 15
output_path = "results/rendezvous.gif"
trail_length = 40  # frames of trail shown behind each point

frame_idx = np.linspace(0, len(t_num) - 1, n_frames).astype(int)

SAT_COLOR = "#2a78d6"
TGT_COLOR = "#eb6834"
GRID_COLOR = "#e0e0e0"
INK_MUTED = "#777777"

fig, (ax_in, ax_rel) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})


def style(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Left: inertial overview (both points overlap at this scale, shows where in the orbit we are)
earth = plt.Circle((0, 0), constants.radius_earth, color="#cde2fb", zorder=0)
ax_in.add_patch(earth)
ax_in.plot(X_tgt[0], X_tgt[1], color=INK_MUTED, linewidth=0.8, linestyle="--", label="Target path")
lim = 1.15*np.max(np.linalg.norm(X_tgt[0:2], axis=0))
ax_in.set_xlim(-lim, lim)
ax_in.set_ylim(-lim, lim)
ax_in.set_aspect("equal")
ax_in.set_xlabel("X [km]")
ax_in.set_ylabel("Y [km]")
ax_in.set_title("Inertial frame", loc="left")
style(ax_in)
tgt_in, = ax_in.plot([], [], "o", color=TGT_COLOR, markersize=9, label="Target")
sat_in, = ax_in.plot([], [], "o", color=SAT_COLOR, markersize=5, label="Satellite")
ax_in.legend(loc="upper right", frameon=False, fontsize=8)

# Right: target-centered LVLH close-up (along-track horizontal, radial vertical)
# Square window (same scale on both axes) centered on the motion
x_lo, x_hi = min(along.min(), 0), max(along.max(), 0)
y_lo, y_hi = min(radial.min(), 0), max(radial.max(), 0)
half = 0.6*max(x_hi - x_lo, y_hi - y_lo, 1.0)
x_c, y_c = (x_lo + x_hi)/2, (y_lo + y_hi)/2
ax_rel.set_xlim(x_c - half, x_c + half)
ax_rel.set_ylim(y_c - half, y_c + half)
ax_rel.set_aspect("equal")
ax_rel.set_xlabel("Along-track [m]")
ax_rel.set_ylabel("Radial [m]")
ax_rel.set_title("Relative to target (LVLH)", loc="left")
style(ax_rel)
ax_rel.plot(0, 0, "o", color=TGT_COLOR, markersize=10, label="Target", zorder=3)
trail, = ax_rel.plot([], [], color=SAT_COLOR, linewidth=1.5, alpha=0.5)
sat_rel, = ax_rel.plot([], [], "o", color=SAT_COLOR, markersize=8, label="Satellite", zorder=4)
ax_rel.legend(loc="upper right", frameon=False, fontsize=8)
info = ax_rel.text(0.03, 0.97, "", transform=ax_rel.transAxes, fontsize=9, family="monospace",
                   va="top", color="#333333")


def update(i):
    k = frame_idx[i]
    k0 = frame_idx[max(0, i - trail_length)]

    tgt_in.set_data([X_tgt[0, k]], [X_tgt[1, k]])
    sat_in.set_data([X_sat[0, k]], [X_sat[1, k]])

    trail.set_data(along[k0:k + 1], radial[k0:k + 1])
    sat_rel.set_data([along[k]], [radial[k]])

    dist = np.linalg.norm(rho[:, k])
    info.set_text(f"t = {t_num[k]/60:5.1f} min\n|ρ| = {dist:7.2f} m\ncross = {cross[k]:6.2f} m")

    return tgt_in, sat_in, trail, sat_rel, info


anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
plt.tight_layout()
anim.save(output_path, writer=PillowWriter(fps=fps))
print(f"Saved {output_path}")
