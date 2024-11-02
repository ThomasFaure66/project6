import numpy as np
from numpy import array

m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)

def solve_verlet(H_pos0 : array,
                 Cl_pos0 : array,
                 F : callable, 
                 dt : float, 
                 N : int, 
                 H_vel0 = None,
                 Cl_vel0 = None):
    
    # durée : N*dt
    # N+1 points
    
    if H_vel0 is None:
        H_vel0 = np.zeros(3)
    if Cl_vel0 is None:
        Cl_vel0 = np.zeros(3)

    # initialisation des vitesses    
    H_pos, Cl_pos = [H_pos0], [Cl_pos0]
    H_vel, Cl_vel = [H_vel0, H_vel0], [Cl_vel0, Cl_vel0]

    # première itération avec potentiellement vitesse aleatoire
    force = F(H_pos[-1], Cl_pos[-1])*(dt**2)
    H_pos1 = H_pos0 + H_vel0*dt  + force/m_H
    Cl_pos1 = Cl_pos0 + Cl_vel0*dt 
    H_pos.append(H_pos1)
    Cl_pos.append(Cl_pos1)

    # Le reste des itérations
    for k in range(N-1):
        force = F(H_pos[-1], Cl_pos[-1])*(dt**2)
        Hk = 2*H_pos[-1] - H_pos[-2] + force/m_H
        Clk = 2*Cl_pos[-1] - Cl_pos[-2] - force/m_Cl

        vClk = (3*Clk - 4*Cl_pos[-1] + Cl_pos[-2])/(2*dt)
        vHk = (3*Hk - 4*H_pos[-1] + H_pos[-2])/(2*dt)

        H_pos.append(Hk.copy())
        Cl_pos.append(Clk.copy())
        H_vel.append(vHk.copy())
        Cl_vel.append(vClk.copy())

    return H_pos, Cl_pos, H_vel, Cl_vel


