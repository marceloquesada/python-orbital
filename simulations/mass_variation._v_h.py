import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.GetClassicOrbitalElements import *
from utils.visualization import plot_classic_orbital_elements


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 200000, 1000000)
earth_radius = 6378.0  # in km
mu = 3.986e5
thrust = 1.1e-3  # N (BIT-3 ~1.1 mN)

# ===================== ADIÇÕES (massa e Busek BIT-3) =====================
# Dados do BIT-3
T   = 1.1e-3       # N (força fixa)
Isp = 2150.0       # s
g0  = 9.80665      # m/s^2

# Estado de massa
m_sat = 20.0       # kg (massa inicial nominal usada no seu aV/aH)
m0    = 20.0       # kg (massa inicial do estado)
m_dry = 15.0       # kg (massa seca)  << já existia e mantida
# ========================================================================

# ===================== ADIÇÕES (H e janelas em anomalia verdadeira) =====================
# OBS: aV/aH fixos foram abandonados; usamos a_inst = (T/m)/1000 dentro de x_dot
THRUST_INTERVAL_DEG = 10.0
MEAN_THETA_LIST_DEG = [180.0]

def throttle(t, x):
    # Sempre ligado enquanto m > m_dry
    return 1.0 if x[6] > m_dry else 0.0

def wrap_deg(a):
    """Normaliza ângulo para [0, 360)."""
    return np.remainder(a, 360.0)

def angle_in_window_deg(theta_deg, center_deg, width_deg):
    """Retorna True se theta estiver dentro da janela centrada em center_deg com largura width_deg."""
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

def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]

    # Gravidade
    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec)
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # Normas e direção
    vnorm = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)

    # ---------- PATCH: aceleração e empuxo só com propelente disponível ----------
    m_cur = max(x[6], 1e-18)                 # kg (evita div/0)
    u = throttle(t, x)                       # 1.0 se m > m_dry, senão 0.0
    a_inst = (T / m_cur) / 1000.0            # km/s^2 (força fixa T, massa variável)

    # (opcional) Empuxo tangencial V -> no seu original, sempre ligado
    if u > 0.0 and vnorm > 1e-12:            # só aplica se houver propelente
        v_hat = v_vec / vnorm
        xdot[3:6] += a_inst * v_hat

    # Empuxo H: liga se θ estiver em qualquer janela configurada
    fire_H = False
    if h_norm > 1e-12:
        theta_deg = get_true_anomaly(r_vec, v_vec, mu)  # usa utilitário do módulo
        fire_H = in_any_window(theta_deg)
        if u > 0.0 and fire_H:               # só aplica se houver propelente
            h_dir = h_vec / h_norm
            xdot[3:6] += a_inst * h_dir

    # Consumo de propelente (só quando há propelente **e** algum modo ativo)
    thr_on = 1.0 if (u > 0.0 and ((vnorm > 0.0) or fire_H)) else 0.0
    xdot[6] = - thr_on * (T/(Isp*g0))
    # ---------------------------------------------------------------------------

    return xdot

# >>> Estado inicial inclui massa <<<
x0 = np.concatenate((r, v, [m0]))

sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method='RK45')

X = sol.y

orbital_elementss = []
nus_deg = []
incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    # anomalia verdadeira (graus) usando utilitário
    nus_deg.append(get_true_anomaly(r_vec, v_vec, mu))
    # inclinação via utilitário
    incs_deg.append(get_inclination(r_vec, v_vec))

plt.figure()
ax = plt.axes(projection='3d')
u, vgrid = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="r")
ax.plot3D(X[0, :], X[1, :], X[2, :], 'b-')
ax.set_title('Orbit Propagation')
ax.axis('equal')
plt.show()

plt.figure()
plt.plot(nus_deg, incs_deg, '.', ms=1.0)
plt.xlim(0, 360)
plt.xlabel('Anomalia Verdadeira, ν (graus)')
plt.ylabel('Inclinação, i (graus)')
plt.title('Inclinação vs Anomalia Verdadeira')
plt.grid(True, alpha=0.3)
plt.show()

# (opcional) massa vs tempo — útil para checar consumo
plt.figure()
plt.plot(t, X[6, :])
plt.xlabel('Tempo [s]'); plt.ylabel('Massa [kg]')
plt.title('Consumo de Propelente')
plt.grid(True, alpha=0.3)
plt.show()

plot_classic_orbital_elements(t, orbital_elementss)
