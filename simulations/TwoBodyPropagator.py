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
thrust = 1.1e-3
def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]
    h= np.cross(x[0:3], x[3:6])
    h_hat=np.linalg.norm(h)
    xdot[3:6] = -(mu/(np.linalg.norm(x[0:3]))**3)*x[0:3] + (thrust*np.linalg.norm(xdot[0:3]))

    return xdot

x0 = np.concatenate((r, v)) 
sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method='RK45')

X = sol.y

orbital_elementss = []
for i in range(X.shape[1]):
    x = X[:, i]
    orbital_elementss.append(get_orbital_elements(x[0:3], x[3:6], mu))

plt.figure()
ax = plt.axes(projection='3d')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = earth_radius * np.cos(u)*np.sin(v)
y = earth_radius * np.sin(u)*np.sin(v)
z = earth_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="r")
ax.plot3D(X[0, :], X[1, :], X[2, :], 'b-')
ax.set_title('Orbit Propagation')
ax.axis('equal')
plt.show()

plot_classic_orbital_elements(t, orbital_elementss)





