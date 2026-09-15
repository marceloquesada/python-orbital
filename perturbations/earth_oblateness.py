import numpy as np
from orbital_utils import constants


class J2_perturbation():

    def get_acceleration(self, state_vector, dt):
        r = np.linalg.norm(state_vector[0:3])
        v = np.linalg.norm(state_vector[3:6])

        pert_vec = np.array([state_vector[0]*(5*((state_vector[2]**2)/(r**2)) - 1),
                             state_vector[1]*(5*((state_vector[2]**2)/(r**2)) - 1),
                             state_vector[2]*(5*((state_vector[2]**2)/(r**2)) - 3),
                           ], dtype=np.float128)
        
        a_j2 = ((3*constants.mu_earth*constants.j2_earth*(constants.radius_earth**2))/(2*(r**5))) * pert_vec

        return a_j2
    