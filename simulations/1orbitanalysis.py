import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.GetClassicOrbitalElements import *
from utils.visualization import plot_classic_orbital_elements


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
earth_radius = 6378.0  # in km
mu = 3.986e5
thrust = 1.1e-3

# ===================== período osculante (1 órbita) usando utilitários =====================
a0 = get_major_axis(r, v, mu)
e0 = get_eccentricity(r, v, mu)
T_orb = 2.0*np.pi*np.sqrt(a0**3/mu)           # período osculante (s)
t = np.linspace(0.0, T_orb, 1_000_000)        # 1 órbita

# ===================== ADIÇÕES (H e janelas em anomalia verdadeira) =====================
# Acelerações efetivas (defina direto em km/s^2 se quiser "massa desprezível")
m_sat = 20.0
aV = (thrust/m_sat)/1000.0   # km/s^2, empuxo tangencial V (deixe ligado p/ espiral)
aH = (thrust/m_sat/1000.0)   # km/s^2, empuxo em H (normal ao plano) durante a janela

# Parâmetros das janelas de anomalia verdadeira (em graus):
THRUST_INTERVAL_DEG = 10.0
MEAN_THETA_LIST_DEG = [180.0]   # ex.: apogeu

def wrap_deg(a):
    """Normaliza ângulo para [0, 360). Versão vetorizável."""
    return np.remainder(a, 360.0)

def angle_in_window_deg(theta_deg, center_deg, width_deg):
    """True se theta estiver dentro da janela centrada em center_deg com largura width_deg."""
    half = 0.5*width_deg
    lo = wrap_deg(center_deg - half)
    hi = wrap_deg(center_deg + half)
    th = wrap_deg(theta_deg)
    if lo <= hi:
        return (th >= lo) and (th <= hi)
    else:
        # janela cruza 0/360
        return (th >= lo) or (th <= hi)

def in_any_window(theta_deg):
    """Combina a lista de janelas e decide o firing."""
    return any(angle_in_window_deg(theta_deg, cdeg, THRUST_INTERVAL_DEG)
               for cdeg in MEAN_THETA_LIST_DEG)

# ========================================================================================

def x_dot(ti, x):
    xdot = np.zeros_like(x)
    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]

    # Gravidade
    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec)
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # (opcional) Empuxo tangencial V para espiral (pode comentar estas linhas se quiser V=off)
    vnorm = np.linalg.norm(v_vec)
    if vnorm > 1e-12:  # tolerância numérica
        v_hat = v_vec / vnorm
        xdot[3:6] += aV * v_hat

    # Empuxo H: liga se θ estiver em qualquer janela configurada
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm > 1e-12:  # tolerância numérica
        theta_deg = get_true_anormaly(r_vec, v_vec, mu)  # usa utilitário
        if in_any_window(theta_deg):
            h_dir = h_vec / h_norm
            xdot[3:6] += aH * h_dir

    return xdot

x0 = np.concatenate((r, v)) 
sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method='RK45')

X = sol.y

# ---------- elementos, anomalia verdadeira e inclinação numa passada ----------
orbital_elementss = []
nus_deg = []
incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    # anomalia verdadeira (graus) usando utilitário — ajusta para (180,360) quando r·v < 0
    nu = get_true_anormaly(r_vec, v_vec, mu)        # retorna [0,180]
    if np.dot(r_vec, v_vec) < 0.0:                  # metade descendente
        nu = (360.0 - nu) % 360.0
    nus_deg.append(nu)
    # inclinação via utilitário
    incs_deg.append(get_inclination(r_vec, v_vec, mu))

# ---------- plot 3D da órbita ----------
plt.figure()
ax = plt.axes(projection='3d')
uu, vv = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = earth_radius * np.cos(uu)*np.sin(vv)
y_sph = earth_radius * np.sin(uu)*np.sin(vv)
z_sph = earth_radius * np.cos(vv)
ax.plot_wireframe(x_sph, y_sph, z_sph, color="r")
ax.plot3D(X[0, :], X[1, :], X[2, :], 'b-')
ax.set_title('Orbit Propagation')
ax.axis('equal')
plt.show()

# ---------- i vs ν ----------
plt.figure()
plt.plot(nus_deg, incs_deg, '.', ms=1.0)
plt.xlim(0, 360)
plt.xlabel('Anomalia Verdadeira, ν (graus)')
plt.ylabel('Inclinação, i (graus)')
plt.title('Inclinação vs Anomalia Verdadeira')
plt.grid(True, alpha=0.3)
plt.show()

# ---------- análise 1 órbita (já integramos de 0 a T_orb) ----------
tt = sol.t
X1 = X
thetas1 = nus_deg

fire_mask1 = np.array([in_any_window(th) for th in thetas1], dtype=bool)

dt = np.diff(tt)
t_H_on = float(np.sum(dt * fire_mask1[:-1].astype(float)))  # [s]

# Δv_H acumulado com aH constante deste arquivo
delta_v_H_kms = aH * t_H_on           # [km/s]
delta_v_H_ms  = 1000.0 * delta_v_H_kms  # [m/s]

# ---------- v_apo direto da trajetória (SEM forma analítica) com suavização local ----------
r_norm_series = np.linalg.norm(X1[0:3, :].T, axis=1)
k_peak = int(np.argmax(r_norm_series))  # índice do maior raio
N = 50                                  # meia-janela para média local (ajuste se precisar)
i0 = max(0, k_peak - N)
i1 = min(X1.shape[1], k_peak + N + 1)
v_apo = np.mean(np.linalg.norm(X1[3:6, i0:i1], axis=0))  # [km/s] média local em torno do apogeu

# Δi_ideal (aproximação instantânea usando v_apo numérico)
arg = np.clip(delta_v_H_kms/(2.0*v_apo), -1.0, 1.0)
delta_i_ideal_deg = np.degrees(2.0*np.arcsin(arg))

# Δi simulado
delta_i_sim_deg = incs_deg - incs_deg[0]

# ---------- gráfico comparativo ----------
plt.figure(figsize=(8,4))
plt.plot(tt/60.0, delta_i_sim_deg, label='Δi simulado (1 órbita)', lw=1.0)
plt.axhline(delta_i_ideal_deg, linestyle='--', label=f'Δi ideal (apogeu) = {delta_i_ideal_deg:.6f}°')
plt.xlabel('Tempo na órbita [min]')
plt.ylabel('Δi [graus]')
plt.title('Mudança de inclinação: Simulação × Analítico (apogeu)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

print("\n=== Analítico × Simulado (1 órbita) ===")
print(f"Período orbital (s):        {T_orb:.3f}")
print(f"Tempo com H ligado (s):     {t_H_on:.6f}")
print(f"Δv_H acumulado (m/s):       {delta_v_H_ms:.6f}")
print(f"v_apogeu (km/s):            {v_apo:.9f}")
print(f"Δi_ideal (graus):           {delta_i_ideal_deg:.9f}")
print(f"Δi_sim (último - inicial):  {delta_i_sim_deg[-1]:.9f}")

# ---------- elementos clássicos (visual) ----------
plot_classic_orbital_elements(t, orbital_elementss)
