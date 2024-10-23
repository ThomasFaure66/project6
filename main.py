import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constantes physiques
k = 4.61  # Constante de raideur du ressort (N/m)
r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)
m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)

# Paramètres de simulation

dt = 1e-15  # Pas de temps (s)
n_steps = 100000  # Nombre de pas de temps

# Fonction pour calculer la distance entre les deux atomes
def distance(H_pos, Cl_pos):
    return np.sqrt(np.sum((Cl_pos - H_pos)**2))

# Fonction de la force harmonique en fonction de la distance
def force(H_pos, Cl_pos):
    r = distance(H_pos, Cl_pos)
    magnitude = k * (r - r_eq)
    direction = (Cl_pos - H_pos) / r  # Vecteur directionnel normalisé
    return magnitude * direction

# Algorithme de Verlet pour l'intégration des équations de mouvement en 3D
def verlet_3d():
    # Initialisation des positions et des vitesses dans l'espace 3D
    H_pos = np.array([-r_eq/10, 0.0, 0.0])  # Position initiale de H
    Cl_pos = np.array([r_eq, 0.0, 0.0])  # Position initiale de Cl (à r_eq de H)
    
    H_vel = np.array([0.0, 0.0, 0.0])  # Vitesse initiale de H
    Cl_vel = np.array([0.0, 0.0, 0.0])  # Vitesse initiale de Cl
    
    # Listes pour enregistrer les positions au cours du temps
    H_positions = [H_pos.copy()]
    Cl_positions = [Cl_pos.copy()]
    H_velocity = [H_vel.copy()]
    Cl_velocity = [Cl_vel.copy()]
    time_list = [0]
    
    #Calcul du premier pas : 

    H_pos_new = H_pos
    Cl_pos_new = Cl_pos 
    H_pos = H_pos_new
    Cl_pos = Cl_pos_new
    H_vel_new = H_vel
    Cl_vel_new = Cl_vel

    H_positions.append(H_pos.copy())
    Cl_positions.append(Cl_pos.copy())
    H_velocity.append(H_vel.copy())
    Cl_velocity.append(Cl_vel.copy())

    #Calcul du deuxième pas : 

    F = force(H_pos, Cl_pos)
    H_acc = F / m_H  # Accélération de H
    Cl_acc = -F / m_Cl

    H_pos_new = 2*H_pos - H_positions[-2] + H_acc * dt**2
    Cl_pos_new = 2*Cl_pos - Cl_positions[-2] + Cl_acc * dt**2

    H_pos = H_pos_new
    Cl_pos = Cl_pos_new

    H_positions.append(H_pos.copy())
    Cl_positions.append(Cl_pos.copy())
    H_velocity.append(H_vel.copy())
    Cl_velocity.append(Cl_vel.copy())

    for step in range(1, n_steps-2):
        # Calcul des forces
        F = force(H_pos, Cl_pos)
        
        # Mise à jour des positions via l'algorithme de Verlet
        H_acc = F / m_H  # Accélération de H
        Cl_acc = -F / m_Cl  # Accélération de Cl (force opposée)
        
        # Mise à jour des positions
        H_pos_new = 2*H_pos - H_positions[-2] + H_acc * dt**2
        Cl_pos_new = 2*Cl_pos - Cl_positions[-2] + Cl_acc * dt**2
        H_vel_new = (3*H_pos - 4*H_positions[-2]+H_positions[-3])/(2*dt)
        Cl_vel_new = (3*Cl_pos - 4*Cl_positions[-2]+Cl_positions[-3])/(2*dt)
        # Enregistrement des positions et du temps
        H_pos = H_pos_new
        Cl_pos = Cl_pos_new
        H_vel = H_vel_new
        Cl_vel = Cl_vel_new
        H_positions.append(H_pos.copy())
        Cl_positions.append(Cl_pos.copy())
        H_velocity.append(H_vel.copy())
        Cl_velocity.append(Cl_vel.copy())
        time_list.append(step * dt)
    
    return time_list, H_positions, Cl_positions, H_velocity, Cl_velocity

# Lancement de la simulation

time, H_positions, Cl_positions, H_velocity, Cl_velocity = verlet_3d()
# Conversion des résultats pour la visualisation
H_positions = np.array(H_positions)
Cl_positions = np.array(Cl_positions)
H_velocity = np.array(H_velocity)
Cl_velocity = np.array(Cl_velocity)

v_rel = H_velocity-Cl_velocity
v_rel_norm = np.array([np.linalg.norm(v_rel[t]) for t in range(n_steps)])
µ = (m_H * m_Cl)/(m_H + m_Cl)
energie_vibratoire = .5*µ*(v_rel_norm**2)

r = (H_positions-Cl_positions)
dist = np.array([np.linalg.norm(r[t]) for t in range(n_steps)])
energie_potentielle = .5*k*((dist-r_eq)**2)

# #Visualisation de la distance entre H et Cl au cours du temps
# distances = [distance(H_positions[i], Cl_positions[i]) for i in range(0,8000)]
# temps = time[0:8000]
# plt.figure(figsize=(10, 6))
# plt.plot(temps, distances)
# plt.axhline(r_eq, color='red', linestyle='--', label='Distance d\'équilibre')
# plt.xlabel('Temps (s)')
# plt.ylabel('Distance H-Cl (m)')
# plt.title('Distance entre H et Cl au cours du temps')
# plt.legend()
# plt.grid(True)
# plt.show()

# #Visualtion de la position sur x de H et de Cl au cours du temps
# position_H = [H_positions[i][0] for i in range(0,8000)]
# position_Cl = [Cl_positions[i][0] for i in range(0,8000)]
# temps = time[0:8000]
# plt.figure(figsize=(10,6))
# plt.plot(temps, position_H)
# plt.plot(temps, position_Cl)
# plt.xlabel('Temps (s)')
# plt.grid(True)
# plt.show()

# #Visualisation de la vitesse sur x de Cl au cours du temps 
# vitesse_Cl = [Cl_velocity[i][0] for i in range(0,8000)]
# temps = time[0:8000]
# plt.figure(figsize=(10,6))
# plt.plot(temps, vitesse_Cl)
# plt.xlabel('Temps (s)')
# plt.grid(True)
# plt.show()

# Visualisation de la vitesse sur x de Cl au cours du temps 
E = energie_vibratoire + energie_potentielle
plot1  = [energie_vibratoire[i] for i in range(8000)]
plot2 = [energie_potentielle[i] for i in range(8000)]
plot3 = [E[i] for i in range(8000)]
temps = time[0:8000]
plt.figure(figsize=(10,6))
plt.plot(temps, plot1, c="red")
plt.plot(temps, plot2, c="blue")
plt.plot(temps, plot3, c="green")
plt.grid(True)
plt.show()