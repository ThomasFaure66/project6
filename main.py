import numpy as np
from solver import solve_verlet

d0 = 127.5e-12
D = 4.6141
alpha = 1.81e-10
omega = np.sqrt(2*D)*alpha
dt = (2*np.pi)/(200*omega)

mH, mCl = 1.67e-27, 58.85e-27

# Initialisation des position et vitesses

rH = np.zeros(3)
rCl = np.array([d0, 0, 0])

R0 = np.stack([rH, rCl])


def F(R):
    
    l = R[0] - R[1]
    F0 = - omega**2


