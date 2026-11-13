"""
Unperturbed (two-body) orbit vs. orbit perturbed by atmospheric drag (IPS density model) + J2.

Both orbits are propagated with the same integrator and tolerances. The initial condition of the
two-body orbit is chosen by `initial_condition`:
    "same_state": both orbits start from the same state vector (same osculating elements). The
                  period-averaged elements then start apart by the J2 short-period offset
                  (~10 km in a for a near-polar orbit starting at the equator).
    "same_mean":  the two-body orbit starts from the mean (one-period averaged) elements of the
                  perturbed orbit at epoch, so both start on the same averaged orbit and the element
                  differences show only the secular effects. The position error is then not zero at
                  t = 0, since the instantaneous positions differ by the short-period terms.
The script plots:
    1. the classic orbital elements of both orbits, averaged over one orbital period;
    2. the period-averaged element differences (perturbed - unperturbed) over time;
    3. the position error, total and decomposed in the RSW (radial / along-track / cross-track)
       frame of the unperturbed orbit, plus the velocity error magnitude.

Run from the repository root:
    python -m simulations.2c_vs_perturbed
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp

from orbital_elements import elements, state_vectors, tle
from orbital_utils import constants
from perturbations import atmospheric_drag, earth_oblateness
from propagators import cowell_propagators


# INITIAL ORBIT: [a (km), e, i (deg), RAAN (deg), arg. of perigee (deg), true anomaly (deg)]
state_vector_0 = tle.TLE_to_state_vectors([
    "1 47950U 21022U   23302.87734107  .00025966  00000-0  12101-2 0  9994",
    "2 47950  97.4765 198.9518 0014055 219.6580 140.3628 15.19780099143496"
    ], constants.mu_earth)
oes_0 = elements.get_osculating_elements(state_vector_0, constants.mu_earth)

# SPACECRAFT / ATMOSPHERE (IPS model)
drag_coef = 2.2
mass = 2                # [kg]
area = 0.2*0.1          # Cross-sectional area [m^2]
gamma = area/mass       # Area-to-mass ratio [m^2/kg]
f107 = 150              # Solar radio flux [sfu]
ap = 15                 # Geomagnetic Ap index

# PROPAGATION
t_days = 1000
t_step = 100            # Output grid and maximum integrator step [s]; also the element sampling
reentry_altitude = 100  # Stop the perturbed orbit below R_earth + this [km]
integration_method = "Radau"
initial_condition = "same_mean"  # "same_mean" or "same_state" (see module docstring)
solver_options = {"rtol": 1e-9, "atol": 1e-9}

ELEMENT_LABELS = ["a [km]", "e [-]", "i [°]", "Ω [°]", "ω [°]", "θ [°]"]
ANGLE_ROWS = [2, 3, 4, 5]


def to_elements(X):
    return np.array([elements.get_osculating_elements(X[:, k], constants.mu_earth)
                     for k in range(X.shape[1])]).T


def unwrap_degrees(angle):
    return np.rad2deg(np.unwrap(np.deg2rad(angle)))


def period_average(t, oes):
    """
    Centred moving average of the elements over one orbital period. The window at each sample is the
    local period T = 2*pi*sqrt(a^3/mu) from the osculating a, so it follows the period of a decaying orbit.
    Angles must be unwrapped beforehand. Samples whose window runs past the data are set to NaN.
    """
    T = 2*np.pi*np.sqrt(oes[0]**3/constants.mu_earth)
    lo = np.searchsorted(t, t - T/2, side="left")
    hi = np.searchsorted(t, t + T/2, side="right")
    valid = (t - T/2 >= t[0]) & (t + T/2 <= t[-1])

    offset = oes[:, :1]  # Reduce cumulative-sum round-off for large unwrapped angles
    csum = np.concatenate([np.zeros((oes.shape[0], 1)), np.cumsum(oes - offset, axis=1)], axis=1)
    avg = (csum[:, hi] - csum[:, lo])/(hi - lo) + offset
    avg[:, ~valid] = np.nan
    return avg


def mean_elements_at_epoch(state_vector, perturbations, samples_per_half_period=500):
    """
    Mean elements at t = 0: the elements of the perturbed orbit averaged over one orbital period centred
    on the epoch (propagated half a period backward and forward). Same averaging as period_average /
    averaged_elements: the eccentricity vector and the argument of latitude are averaged for ω and θ.
    Returns [a, e, i, Ω, ω, θ] with angles in degrees.
    """
    a0 = elements.get_major_axis(state_vector[0:3], state_vector[3:6], constants.mu_earth)
    half = np.pi*np.sqrt(a0**3/constants.mu_earth)

    def x_dot(t, x):
        acc = -constants.mu_earth/np.linalg.norm(x[0:3])**3*x[0:3]
        for perturbation in perturbations:
            acc = acc + perturbation.get_acceleration(x, t)
        return np.concatenate([x[3:6], acc])

    arcs = []
    for t_end in (-half, half):
        t_eval = np.linspace(0, t_end, samples_per_half_period + 1)
        arcs.append(solve_ivp(x_dot, (0, t_end), state_vector, t_eval=t_eval, method="DOP853",
                              rtol=1e-11, atol=1e-11).y)
    X = np.concatenate([arcs[0][:, :0:-1], arcs[1][:, :-1]], axis=1)  # Uniform samples on [-T/2, T/2)

    a, e, i, raan, argp, theta = to_elements(X)
    w = np.deg2rad(argp)
    ex, ey = np.mean(e*np.cos(w)), np.mean(e*np.sin(w))
    u = np.mean(unwrap_degrees(argp + theta))
    argp_mean = np.rad2deg(np.arctan2(ey, ex))
    return np.array([a.mean(), e.mean(), i.mean(), np.mean(unwrap_degrees(raan)) % 360,
                     argp_mean % 360, (u - argp_mean) % 360])


def rsw_components(X_ref, dr):
    """Project dr onto the radial / along-track / cross-track frame of the reference orbit X_ref."""
    r, v = X_ref[0:3], X_ref[3:6]
    R = r/np.linalg.norm(r, axis=0)
    W = np.cross(r, v, axis=0)
    W /= np.linalg.norm(W, axis=0)
    S = np.cross(W, R, axis=0)
    return np.sum(dr*R, axis=0), np.sum(dr*S, axis=0), np.sum(dr*W, axis=0)


def reentry(t, x):
    return np.linalg.norm(x[0:3]) - (constants.radius_earth + reentry_altitude)
reentry.terminal = True

t = np.arange(0, t_days*86400 + t_step, t_step)

perturbations = [
    atmospheric_drag.Atmospheric_drag_perturbation_ips(drag_coef, gamma, f107, ap, rotating_atmosphere=True),
    earth_oblateness.J2_perturbation(),
]

# PERTURBED: IPS drag + J2 (run first: it sets the time span, since it stops at re-entry)
pert_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, perturbations)
t_pert, X_pert = pert_prop.propagate(t, integration_method=integration_method, events=reentry, **solver_options)
pert_prop.progress_bar.close()
t_events = pert_prop.solution.t_events[0]
if t_events.size:
    print(f"Perturbed orbit re-entered ({reentry_altitude} km) at {t_events[0]/86400:.1f} days")

# UNPERTURBED: only over the time span of the perturbed orbit
if initial_condition == "same_mean":
    oes_mean_0 = mean_elements_at_epoch(state_vector_0, perturbations)
    state_vector_unpert_0 = state_vectors.get_state_vectors(oes_mean_0, constants.mu_earth)
    print("Osculating elements at epoch:", np.round(oes_0, 6))
    print("Mean elements at epoch:      ", np.round(oes_mean_0, 6))
elif initial_condition == "same_state":
    state_vector_unpert_0 = state_vector_0
else:
    raise ValueError(f"Unknown initial_condition: {initial_condition}")

unpert_prop = cowell_propagators.Unperturbed_propagator(state_vector_unpert_0, constants.mu_earth)
t_unpert, X_unpert = unpert_prop.propagate(t[:len(t_pert)], integration_method=integration_method,
                                           **solver_options)
unpert_prop.progress_bar.close()

# Align lengths (safety: both should already match)
n = min(X_unpert.shape[1], X_pert.shape[1])
t_s = t[:n]
t_d = t_s/86400
X_unpert, X_pert = X_unpert[:, :n], X_pert[:, :n]

# Period-averaged elements
def averaged_elements(t, X):
    """
    Average the elements over one orbital period. For a near-circular orbit the osculating ω and θ jump by
    ~180° in opposite directions, so they are not averaged directly: the eccentricity vector components
    (e cos ω, e sin ω) and the argument of latitude u = ω + θ are averaged instead, then ω and θ are rebuilt.
    Returned angles are unwrapped (continuous).
    """
    a, e, i, raan, argp, theta = to_elements(X)
    w = np.deg2rad(argp)
    smooth = np.array([a, e, i, unwrap_degrees(raan), e*np.cos(w), e*np.sin(w), unwrap_degrees(argp + theta)])
    a, e, i, raan, ex, ey, u = period_average(t, smooth)

    argp = np.full_like(a, np.nan)
    valid = ~np.isnan(a)
    argp[valid] = unwrap_degrees(np.rad2deg(np.arctan2(ey[valid], ex[valid])))
    return np.array([a, e, i, raan, argp, u - argp])


avg_unpert = averaged_elements(t_s, X_unpert)
avg_pert = averaged_elements(t_s, X_pert)

d_oes = avg_pert - avg_unpert  # Angles kept unwrapped: continuous drift
last = np.where(~np.isnan(d_oes[0]))[0][-1]

dr = X_pert[0:3] - X_unpert[0:3]
dv = X_pert[3:6] - X_unpert[3:6]
pos_err = np.linalg.norm(dr, axis=0)
vel_err = np.linalg.norm(dv, axis=0)
radial, along, cross = rsw_components(X_unpert, dr)

print(f"After {t_d[-1]:.1f} days:")
print(f"  position error: {pos_err[-1]:.3f} km (max {pos_err.max():.3f} km)")
print(f"  velocity error: {vel_err[-1]*1e3:.3f} m/s")
print(f"Period-averaged differences at {t_d[last]:.1f} days:")
print(f"  Δa = {d_oes[0, last]*1e3:.1f} m, ΔΩ = {d_oes[3, last]:.4f}°, Δω = {d_oes[4, last]:.4f}°, "
      f"Δθ = {d_oes[5, last]:.1f}°")
avg_unpert[3:6] %= 360  # Elements plot shows angles in [0, 360)
avg_pert[3:6] %= 360

title = (f"Dois corpos vs. arrasto IPS + J2 — h₀ = {oes_0[0] - constants.radius_earth:.0f} km, "
         f"i = {oes_0[2]:.4f}°, $C_D$ = {drag_coef}, A/m = {gamma:g} m²/kg, F10.7 = {f107}, Ap = {ap}\n"
         + ("Condição inicial: mesmos elementos médios" if initial_condition == "same_mean"
            else "Condição inicial: mesmo vetor de estado"))

# 1. ORBITAL ELEMENTS
fig1, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
for k, ax in enumerate(axes.flat):
    ax.plot(t_d, avg_unpert[k], lw=0.8, label="Dois corpos")
    ax.plot(t_d, avg_pert[k], lw=0.8, label="Arrasto (IPS) + J2")
    ax.set_ylabel(ELEMENT_LABELS[k])
    ax.grid(True, alpha=0.3)
axes[0, 0].legend()
for ax in axes[-1]:
    ax.set_xlabel("Tempo [dias]")
fig1.suptitle("Elementos orbitais médios em um período\n" + title, fontsize=10)
fig1.tight_layout()

# 2. ELEMENT DIFFERENCES
fig2, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
for k, ax in enumerate(axes.flat):
    ax.plot(t_d, d_oes[k], lw=0.8, color="tab:red")
    ax.set_ylabel("Δ" + ELEMENT_LABELS[k])
    ax.grid(True, alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("Tempo [dias]")
fig2.suptitle("Diferença dos elementos médios em um período (perturbado − dois corpos, ângulos contínuos)\n" + title, fontsize=10)
fig2.tight_layout()

# 3. DISTANCE ERRORS
fig3, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axes[0].plot(t_d, pos_err, lw=0.8, color="k")
axes[0].set_ylabel("|Δr| [km]")
axes[1].plot(t_d, radial, lw=0.8, label="Radial")
axes[1].plot(t_d, along, lw=0.8, label="Ao longo da trajetória")
axes[1].plot(t_d, cross, lw=0.8, label="Fora do plano")
axes[1].set_ylabel("Δr (RSW) [km]")
axes[1].legend()
axes[2].plot(t_d, vel_err*1e3, lw=0.8, color="tab:purple")
axes[2].set_ylabel("|Δv| [m/s]")
axes[2].set_xlabel("Tempo [dias]")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig3.suptitle("Erro de posição e velocidade (perturbado − dois corpos)\n" + title, fontsize=10)
fig3.tight_layout()

os.makedirs("results", exist_ok=True)
fig1.savefig("results/unperturbed_vs_drag_j2_elements.png", dpi=150)
fig2.savefig("results/unperturbed_vs_drag_j2_element_diff.png", dpi=150)
fig3.savefig("results/unperturbed_vs_drag_j2_distance_error.png", dpi=150)
plt.show()
