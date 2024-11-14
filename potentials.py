import numpy as np
from math import exp
import matplotlib.pyplot as plt

D = 4.6141 * 1.6e-19
alpha = 1.81e10
k = 2*D*alpha**2
kB = 1.380e-23 # Constante de Boltzmann

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
    fact = np.exp(-alpha*(r-r_eq))

    return -2*D*alpha*fact*(fact-1) * direction

def pot_har(x):
    return .5*k*x**2

def pot_morse(x):
    return D*(np.exp(-alpha*x)-1)**2

if __name__ == "__main__":
    
    T1 = 100
    T2 = 10000
    x_values = np.linspace(-0.3e-10, 1.3e-10, 10000)
    y_values = pot_morse(x_values)
    z_values = pot_har(x_values)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=r"$Morse$", color="blue")
    plt.plot(x_values, z_values, label =r"$Harmonique$", color="red")
    plt.axhline(.5*kB*T1, color='green', linestyle='--', label=r"100K")
    plt.axhline(.5*kB*T2, color='orange', linestyle='--', label=r"10000K")

    plt.xlabel("r")
    plt.ylabel("Potentiel")
    plt.title("Potentiel")
    plt.legend()
    plt.grid(True)
    plt.show()