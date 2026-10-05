import csv
import numpy as np


def save_orbit(ts, Xs, filename="orbit_1"):
    print(np.array([np.transpose(ts)]).shape)
    print(Xs.shape)

    data_matrix = np.transpose(
        np.vstack((
        np.array([np.transpose(ts)]),
        Xs)))

    np.savetxt(filename+".csv", data_matrix, delimiter=",", fmt='%.10f',
                header="Time (s), R_x (Km), R_y (Km), R_z (Km), V_x (Km/s), V_y (Km/s), V_z (Km/s)",
                comments='')