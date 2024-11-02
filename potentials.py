import numpy as np
from math import exp
import matplotlib.pyplot as plt

D = 4.6141 * 1.6e-19
alpha = 1.81e10
k = 2*D*alpha**2
print(k)
r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)

# Fonction pour calculer la distance entre les deux atomes
def distance(H_pos, Cl_pos):
    return np.linalg.norm(H_pos-Cl_pos)

# Fonction de la force harmonique en fonction de la distance
def force_harmonique(H_pos, Cl_pos):
    r = distance(H_pos, Cl_pos) 
    magnitude = k * (r - r_eq)
    direction = (Cl_pos - H_pos) / r  # Vecteur directionnel normalisé
    return magnitude * direction

def force_morse(H_pos, Cl_pos):
    r = distance(H_pos, Cl_pos)
    direction = (Cl_pos - H_pos) / r  # Vecteur directionnel normalisé
    fact = exp(-alpha*(r-r_eq))

    return 2*D*alpha*fact*(fact-1) * direction

def pot_har(x):
    return k*x**2

def pot_morse(x):
    return D*(exp(-alpha*x)-1)**2
