import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.GetClassicOrbitalElements import *
from utils.visualization import plot_classic_orbital_elements

r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 200000, 100000)
earth_radius = 6378.0  # in km
mu = 3.986e5

thetas = np.linspace(0, 2*np.pi, 100000)

h = np.cross(r, v)
e = np.linalg.norm((np.cross(v, h)/mu) - (r/np.linalg.norm(r)))
l = h**2/mu

rs = np.array([])
for theta in thetas:
    np.append(rs, l/(1 + e * np.cos(theta)))

def get_classic_orbital_elements(r, v, mu):
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)  # Vetor nodal
    e_vec = ((np.linalg.norm(v)**2 - mu/np.linalg.norm(r)) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    
    # Inclinação
    i = np.arccos(h[2]/np.linalg.norm(h))
    
    # Longitude do nó ascendente
    if np.linalg.norm(n) > 1e-10:
        Omega = np.arccos(n[0]/np.linalg.norm(n))
        if n[1] < 0:
            Omega = 2*np.pi - Omega
    else:
        Omega = 0
    
    # Argumento do perigeu
    if np.linalg.norm(n) > 1e-10:
        omega = np.arccos(np.dot(n, e_vec)/(np.linalg.norm(n)*e))
        if e_vec[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = 0
    
    # Semi-latus rectum
    p = np.linalg.norm(h)**2/mu
    
    return p, e, i, Omega, omega, e_vec

# Calcular elementos orbitais
p, e, i, Omega, omega, e_vec = get_classic_orbital_elements(r, v, mu)

# Gerar pontos da órbita no plano perifocal
thetas = np.linspace(0, 2*np.pi, 1000)
rs = p / (1 + e * np.cos(thetas))
x_p = rs * np.cos(thetas)
y_p = rs * np.sin(thetas)
z_p = np.zeros_like(thetas)

# Matriz de rotação do sistema perifocal para inercial
R11 = np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(i)
R12 = -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(i)
R13 = np.sin(Omega)*np.sin(i)

R21 = np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(i)
R22 = -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(i)
R23 = -np.cos(Omega)*np.sin(i)

R31 = np.sin(omega)*np.sin(i)
R32 = np.cos(omega)*np.sin(i)
R33 = np.cos(i)

R = np.array([[R11, R12, R13],
              [R21, R22, R23],
              [R31, R32, R33]])

# Aplicar rotação
r_inertial = np.dot(R, np.array([x_p, y_p, z_p]))

# Plotar em 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar órbita
ax.plot(r_inertial[0, :], r_inertial[1, :], r_inertial[2, :], label='Órbita')

# Plotar Terra
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color='blue', alpha=0.3)

# Configurações do gráfico
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Órbita em 3D')
ax.legend()

# Configurar limites dos eixos
max_range = np.max(np.abs(r_inertial)) * 1.2
ax.set_xlim([-max_range, max_range])
ax.axis('equal')

plt.show()

