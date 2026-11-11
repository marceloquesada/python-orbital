"""
Semi-major axis decay under four atmospheric density models:
exponential (US Standard Atmosphere 1976, Curtis), IPS (variable scale height), TD-88 and NRLMSIS (pymsis).

Conditions follow the worked example of the IPS Radio and Space Services article
"Satellite Orbital Decay Calculations" (Kennewell, updated by R. Panwar):
    mass 100 kg, effective area A*C_D = 1 m^2, circular orbit at 300 km, F10.7 = 70, Ap = 0,
    re-entry at 180 km. Published result: "Re-entry after 46 days".
The article gives no inclination or date; an equatorial orbit is used and the epoch only matters
for TD-88 and NRLMSIS.

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


# CONDITIONS (IPS worked example)
mass = 100                       # [kg]
effective_area = 1               # A*C_D [m^2]
drag_coef = 2.2
gamma = effective_area/drag_coef/mass  # Area-to-mass ratio [m^2/kg], so that C_D*gamma = Ae/m = 0.01 m^2/kg
f107 = 70                        # Solar radio flux (daily and mean) [sfu]
ap = 0                           # Geomagnetic Ap index
k_p = 0                          # Kp equivalent to Ap = 0 (used by TD-88)
epoch = datetime.datetime(2019, 3, 21, 15, 0, 0, tzinfo=datetime.timezone.utc).timestamp()  # Not given by IPS
ips_rotating_atmosphere = False  # The IPS article neglects atmospheric rotation; True for consistency with the other models

oes_0 = np.array([constants.radius_earth + 300, 0.0, 0.0, 0.0, 0.0, 0.0])

t_max_days = 100        # Upper bound; each run stops at re-entry
t_step = 50             # Propagator time grid; also the maximum integrator step [s]
t_sample = 600          # Plot sampling [s], multiple of t_step
reentry_altitude = 180  # Stop when the orbital radius drops below R_earth + this [km] (IPS definition)

# IPS article output table: (time [days], height [km]); heights are a - R_e with R_e = 6378 km
IPS_TABLE = np.array([[0.0, 300.0], [10.5, 289.9], [19.5, 279.9], [26.3, 269.9], [31.5, 259.9],
                      [35.4, 249.9], [38.4, 239.8], [40.6, 229.8], [42.3, 219.6], [43.6, 209.1],
                      [44.5, 199.3], [45.2, 189.2], [45.7, 179.5]])
IPS_DECAY_DAYS = 46

# Previous conditions:
#   Curtis Example 10.1 orbit: oes_0 = [6955, 0.052049, 65.1, 340, 58, 332], re-entry at 100 km
#   Thesis LEO case (section 5.2 / Table 21): oes_0 = [R_earth + 700, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
#   gamma = 5, F10.7 = F10.7a = 150, Kp = 4, day 80 at 15 h UTC, re-entry at 100 km (thesis: 51 days)


models = {
    "Exponential (USSA76, Curtis)": atmospheric_drag.Atmospheric_drag_perturbation_exp(drag_coef, gamma),
    "IPS": atmospheric_drag.Atmospheric_drag_perturbation_ips(
        drag_coef, gamma, f107, ap, rotating_atmosphere=ips_rotating_atmosphere)
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

ax.plot(IPS_TABLE[:, 0], 6378 + IPS_TABLE[:, 1],
        label=f"IPS article table (re-entry after {IPS_DECAY_DAYS} d)")

ax.set_xlabel("Time [days]")
ax.set_ylabel("Semi-major axis [km]")
ax.set_title(f"Semi-major axis decay, 300 km circular, $C_D A/m$ = {effective_area/mass} m²/kg, "
             f"F10.7 = {f107}, Ap = {ap}")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("results/drag_models_sma_ips_case.png", dpi=150)
plt.show()
