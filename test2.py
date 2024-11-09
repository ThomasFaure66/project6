import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from math import exp

r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)
m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)
kB = 1.380e-23 # Constante de Boltzmann
T_ther = 10000  # Température thermostat
D = 4.6141*1.6e-19
alpha = 1.81e10
gamma = 0
k=2*D*alpha**2 # Constante de raideur du ressort (N/m)

T=20
L=[]
for i in range(0,3):
        A = np.random.uniform(0, 1)
        B = np.random.uniform(0, 1)
        Xi = np.sqrt(2)*np.cos(2*np.pi*A)*np.sqrt(-np.log(B))
        v0 = np.sqrt(kB*T/m_H)
        L.append(Xi*v0)


M=[]
for i in range(0,3):
        A = np.random.uniform(0, 1)
        B = np.random.uniform(0, 1)
        Xi = np.sqrt(2)*np.cos(2*np.pi*A)*np.sqrt(-np.log(B))
        v0 = np.sqrt(kB*T/m_Cl)
        M.append(Xi*v0)
    
L2=L.copy()
M2=M.copy()
    
for i in range(0,3):
        L[i] = L2[i]-(m_H*L2[i]+m_Cl*M2[i])/(m_H+m_Cl)
        M[i]= M2[i]-(m_H*L2[i]+m_Cl*M2[i])/(m_H+m_Cl)
        

print((m_H*L[2]+m_Cl*M[2])/(m_H+m_Cl))


# print(L, M)
   