from orbital_elements import elements
from orbital_utils import constants
from utils import csv, visualization
import argparse
import numpy as np

parser = argparse.ArgumentParser(
        description="Plot Classic Orbital Elements over time from a csv file."
    )


parser.add_argument("file", type=str, help="Path to csv file to plot.")
parser.add_argument("-d", "--decimation", type=int, help="Decimation rate to shrink sample size.", default=1)

args = parser.parse_args()

ts, Xs = csv.load_orbit("orbit_1.csv")

oes = np.empty_like(Xs)
ts_f = []
dec_counter = 0
j = 0
for i in range(len(ts)):
    if dec_counter >= args.decimation:
        ts_f.append(ts[i])
        oes[:, j] = elements.get_osculating_elements(Xs[:, i], constants.mu_earth)
        dec_counter = 0
        j += 1
    else:
        dec_counter+=1

oes_f = oes[:, :j]

visualization.plot_classic_orbital_elements(ts_f, oes_f)
