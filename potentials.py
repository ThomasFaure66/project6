import numpy as np

D = 4.6141 * 1.6e19
alpha = 1.81e-10
k = 2*D*alpha**2
r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)

# Fonction pour calculer la distance entre les deux atomes
def distance(H_pos, Cl_pos):
    return np.sqrt(np.sum((Cl_pos - H_pos)**2))

# Fonction de la force harmonique en fonction de la distance
def force_harmonique(H_pos, Cl_pos):
    r = distance(H_pos, Cl_pos) 
    magnitude = k * (r - r_eq)
    direction = (Cl_pos - H_pos) / r  # Vecteur directionnel normalisé
    return magnitude * direction

