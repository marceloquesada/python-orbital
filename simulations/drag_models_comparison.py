"""
Semi-major axis decay under three atmospheric density models:
exponential (CIRA-72 table), TD-88 and NRLMSIS (pymsis).

Conditions follow the LEO case of the reference thesis (section 5.2 / Table 21):
    altitude 700 km (a = R_earth + 700), e = 0.0001, i = RAAN = omega = theta = 0.0001 deg,
    C_D = 2.2, F10.7 = F10.7a = 150, Kp = 4, day of year 80, 15 h UTC.

Run from the repository root:
    python -m simulations.drag_models_comparison
"""
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np

from orbital_elements import elements, state_vectors
from orbital_utils import constants
from perturbations import atmospheric_drag
from propagators import cowell_propagators


# CONDITIONS
gamma = 0.5        # Area-to-mass ratio [m^2/kg] (thesis LEO cases: 0.001, 0.01, 0.1, 0.5, 1, 5)
drag_coef = 2.2
f_x = 150        # Daily F10.7 solar flux [sfu]
f_0 = 150        # 81-day mean F10.7 solar flux [sfu]
k_p = 4          # Geomagnetic index
epoch = datetime.datetime(2019, 3, 21, 15, 0, 0, tzinfo=datetime.timezone.utc).timestamp()  # Day 80, 15 h UTC

oes_0 = np.array([constants.radius_earth + 700, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

t_max_days = 524        # Upper bound; each run stops at re-entry
t_step = 100            # Propagator time grid; also the maximum integrator step [s]
t_sample = 600          # Plot sampling [s], multiple of t_step
reentry_altitude = 100  # Stop when the orbital radius drops below R_earth + this [km]

THESIS_DECAY_DAYS = {0.1: 2621, 0.5: 524.5, 1: 262.5, 5: 51}  # Table 21


models = {
    "Exponential (CIRA-72)": atmospheric_drag.Atmospheric_drag_perturbation_exp(drag_coef, gamma),
    "TD-88": atmospheric_drag.Atmospheric_drag_perturbation_td88(drag_coef, gamma, f_x, f_0, k_p, epoch),
    "NRLMSIS 2.1": atmospheric_drag.Atmospheric_drag_perturbation_msis(
        drag_coef, gamma, f_x, f_0, atmospheric_drag.kp_to_ap(k_p), epoch),
}


def reentry(t, x):
    return np.linalg.norm(x[0:3]) - (constants.radius_earth + reentry_altitude)
reentry.terminal = True


state_vector_0 = state_vectors.get_state_vectors(oes_0, constants.mu_earth)
t = np.arange(0, t_max_days*24*3600, t_step)

fig, ax = plt.subplots(figsize=(9, 5))

for name, perturbation in models.items():
    start = time.time()
    num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [perturbation])
    t_num, X_num = num_prop.propagate(t, integration_method="RK45", events=reentry, rtol=1e-9, atol=1e-9)
    num_prop.progress_bar.close()

    events = num_prop.solution.t_events[0]
    t_reentry = events[0] if events.size else None

    # Decimate to t_sample before computing the elements (plot resolution)
    stride = t_sample//t_step
    t_plot, X_plot = t_num[::stride], X_num[:, ::stride]
    sma = np.array([elements.get_major_axis(X_plot[0:3, k], X_plot[3:6, k], constants.mu_earth)
                    for k in range(X_plot.shape[1])])

    if t_reentry is None:
        label = f"{name} (no re-entry in {t_max_days} d)"
    else:
        label = f"{name}: re-entry at {t_reentry/86400:.1f} d"
    print(f"{label}  [{time.time() - start:.0f} s]")

    ax.plot(t_plot/86400, sma, label=label)

if gamma in THESIS_DECAY_DAYS:
    ax.axvline(THESIS_DECAY_DAYS[gamma], color="k", linestyle="--", linewidth=1,
               label=f"Thesis, Table 21: {THESIS_DECAY_DAYS[gamma]} d")

ax.set_xlabel("Time [days]")
ax.set_ylabel("Semi-major axis [km]")
ax.set_title(f"Semi-major axis decay, LEO 700 km, γ = {gamma} m²/kg, $C_D$ = {drag_coef}")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"results/drag_models_sma_gamma{gamma}.png", dpi=150)
plt.show()
