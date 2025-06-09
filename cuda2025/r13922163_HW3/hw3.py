import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def solve_poisson_3d(L, max_iter=10000, tol=1e-4):
    phi = np.zeros((L, L, L))
    rho = np.zeros((L, L, L))

    center = L // 2
    rho[center, center, center] = 1.0

    for iteration in range(max_iter):
        phi_new = np.copy(phi)

        for i in range(1, L-1):
            for j in range(1, L-1):
                for k in range(1, L-1):
                    phi_new[i, j, k] = (1/6) * (
                        phi[i+1, j, k] + phi[i-1, j, k] +
                        phi[i, j+1, k] + phi[i, j-1, k] +
                        phi[i, j, k+1] + phi[i, j, k-1] +
                        rho[i, j, k]
                    )

        diff = np.max(np.abs(phi_new - phi))
        phi = phi_new

        if diff < tol:
            break

    return phi, center

def compute_phi_vs_r(phi, center):
    L = phi.shape[0]
    r_dict = defaultdict(list)

    for i in range(L):
        for j in range(L):
            for k in range(L):
                r = int(round(np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)))
                if r > 0:
                    r_dict[r].append(phi[i, j, k])

    r_values = sorted(r_dict.keys())
    phi_r = [np.mean(r_dict[r]) for r in r_values]

    return r_values, phi_r

# Run for L=16 as a test
L = 16
phi, center = solve_poisson_3d(L)
r_values, phi_r = compute_phi_vs_r(phi, center)
df = pd.DataFrame({"r": r_values, "phi(r)": phi_r})

print(df)
