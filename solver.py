import numpy as np
from numpy import array

m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)
kB = 1.380e-23 # Constante de Boltzmann

def solve_verlet_lang(H_pos0 : array,
                 Cl_pos0 : array,
                 F : callable, 
                 dt : float, 
                 N : int, 
                 T : float,
                 gamma : float = 0):
    
    # durée : N*dt
    # N+1 points

    H_vel0 = np.sqrt(np.array([0.0, .5*kB*T/m_H, 0.0]))
    Cl_vel0 = -np.sqrt(np.array([0.0, .5*kB*T/m_Cl, 0.0]))

    # initialisation des vitesses 
    H_pos, Cl_pos = [H_pos0, H_pos0 + H_vel0*dt], [Cl_pos0, Cl_pos0 + Cl_vel0*dt]
    H_vel, Cl_vel = [H_vel0, H_vel0], [Cl_vel0, Cl_vel0]

    # première itération avec potentiellement vitesse aleatoire

    # Le reste des itérations
    for k in range(N-1):

        RH = np.sqrt(2*m_H*gamma*kB*T/dt)*np.random.normal(np.zeros(3), 1)
        RCl = np.sqrt(2*m_Cl*gamma*kB*T/dt)*np.random.normal(np.zeros(3), 1)
        frH = -gamma*H_vel[-1]
        frCl = -gamma*Cl_vel[-1]

        force = F(H_pos[-1], Cl_pos[-1])

        Hk = 2*H_pos[-1] - H_pos[-2] + (frH + RH/m_H + force/m_H)*(dt**2)
        Clk = 2*Cl_pos[-1] - Cl_pos[-2]  + (frCl + RCl/m_Cl - force/m_Cl)*(dt**2)

        vClk = (3*Clk - 4*Cl_pos[-1] + Cl_pos[-2])/(2*dt)
        vHk = (3*Hk - 4*H_pos[-1] + H_pos[-2])/(2*dt)

        H_pos.append(Hk.copy())
        Cl_pos.append(Clk.copy())
        H_vel.append(vHk.copy())
        Cl_vel.append(vClk.copy())

    return H_pos, Cl_pos, H_vel, Cl_vel


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

def moyenne_cumul(E):
    n = len(E)
    moyennes = [E[0]]
    for k in range(1, n):
        moyennes.append((k*moyennes[-1] + E[k])/(k+1))

    return moyennes