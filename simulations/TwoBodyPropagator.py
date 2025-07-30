import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 20000, 100000)
earth_radius = 6371.0  # in km
mu = 3.986e5

def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]
    xdot[3:5] = -(mu/np.linalg.norm(x[0:2])**3)*x[0:2]

    return xdot

x0 = np.concatenate((r, v))
sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method='Radau')

X = sol.y[0:2, :]

plt.figure()
ax = plt.axes(projection='3d')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = earth_radius * np.cos(u)*np.sin(v)
y = earth_radius * np.sin(u)*np.sin(v)
z = earth_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="r")
ax.plot3D(X[0, :], X[1, :], np.zeros_like(X[0, :]), 'b-')
ax.set_title('Orbit Propagation')
ax.axis('equal')
plt.show()







