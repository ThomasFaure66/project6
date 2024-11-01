import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constantes physiques
k = 4.61  # Constante de raideur du ressort (N/m)
r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)
m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)
kB = 1.380e-23 # Constante de Boltzmann
T_ther = 30 # Température thermostat
gamma = 0
# Paramètres de simulation

dt = 1e-16 # Pas de temps (s)
n_steps = 50000  # Nombre de pas de temps

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
    # Initialisation des position
    # s et des vitesses dans l'espace 3D
    H_pos = np.array([r_eq, 0.0, 0.0])  # Position initiale de H
    Cl_pos = np.array([-r_eq/10, 0.0, 0.0])  # Position initiale de Cl (à r_eq de H)
    
    H_vel = np.array([0.0, 10.0, 0.0])  # Vitesse initiale de H
    Cl_vel = np.array([0.0, 0.0, 0.0])  # Vitesse initiale de Cl
    
    # Listes pour enregistrer les positions au cours du temps
    H_positions = [H_pos, H_pos + H_vel*dt]
    Cl_positions = [Cl_pos, Cl_pos + Cl_vel*dt]
    H_velocity = [H_vel, H_vel]
    Cl_velocity = [Cl_vel, Cl_vel]
    H_positions = [H_pos, H_pos + H_vel*dt]
    Cl_positions = [Cl_pos, Cl_pos + Cl_vel*dt]
    H_velocity = [H_vel, H_vel]
    Cl_velocity = [Cl_vel, Cl_vel]
    time_list = [0]

    #Calcul du deuxième pas : 

    F = force(H_pos, Cl_pos)

    #Calcul force de langevin :
    R_H = np.array([np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1)])
    R_Cl = np.array([np.sqrt((2*m_Cl*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1)])

    H_acc = F / m_H  - gamma * H_velocity[-1] + R_H/m_H # Accélération de H
    Cl_acc = -F / m_Cl  - gamma*Cl_velocity[-1] + R_Cl/m_Cl# Accélération de Cl (force opposée)

    H_pos_new = 2*H_positions[-1] - H_positions[-2] + H_acc * dt**2
    Cl_pos_new = 2*Cl_positions[-1] - Cl_positions[-2] + Cl_acc * dt**2


    H_pos = H_pos_new
    Cl_pos = Cl_pos_new

    H_positions.append(H_pos)
    Cl_positions.append(Cl_pos)
    H_velocity.append(H_vel)
    Cl_velocity.append(Cl_vel)

    for step in range(1, n_steps-2):

        #Calcul force de Langevin
        R_H = np.array([np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1)])
        R_Cl = np.array([np.sqrt((2*m_Cl*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1),np.sqrt((2*m_H*gamma*kB*T_ther)/dt)*np.random.normal(0, 1)])
  
        # Calcul des forces
        F = force(H_pos, Cl_pos)
        
        H_vel_new = (3*H_pos - 4*H_positions[-2]+H_positions[-3])/(2*dt)
        Cl_vel_new = (3*Cl_pos - 4*Cl_positions[-2]+Cl_positions[-3])/(2*dt)

        #Generation de la force aléatoire
        H_acc = F / m_H  - gamma * H_vel_new + R_H/m_H # Accélération de H
        Cl_acc = -F / m_Cl  - gamma*Cl_vel_new + R_Cl/m_Cl# Accélération de Cl (force opposée)
        
        # Mise à jour des positions
        H_pos_new = 2*H_pos - H_positions[-2] + H_acc * dt**2
        Cl_pos_new = 2*Cl_pos - Cl_positions[-2] + Cl_acc * dt**2
        
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


mu = (m_H*m_Cl)/(m_Cl+m_H)
CM_velocity = (m_H*H_velocity+m_Cl*Cl_velocity)/(m_H+m_Cl)
print("CM_velocity", CM_velocity)
Relative_position = Cl_positions - H_positions
Relative_velocity = Cl_velocity-H_velocity

Translation_energy = 0.5 * (m_Cl+m_H)*np.linalg.norm(CM_velocity, axis = 1)**2
print("Energie de translation", Translation_energy)

Vibrational_energy = 0.5*(mu)*np.linalg.norm(Relative_velocity, axis = 1)**2
print("Energie de vibration", Vibrational_energy)

omega = np.linalg.norm(np.cross(Relative_position, Relative_velocity), axis=1) / (np.linalg.norm(Relative_position, axis=1)**2)
I = mu*np.linalg.norm(Relative_position, axis = 1)**2
Rotational_energy = 0.5*I*omega**2
print("energie de rotation", Rotational_energy)

print(Relative_position)
Potential_energy = 0.5*k*((np.linalg.norm(Relative_position, axis = 1)-r_eq)**2)
print("energie potentielle", Potential_energy)

Total_energy = Potential_energy + Translation_energy + Vibrational_energy + Rotational_energy
print("energie totale", Total_energy)

plot1  = [Total_energy[i] for i in range(0,n_steps-2)]
plot2 = [Potential_energy[i] for i in range(0,n_steps-2)]
plot3 = [Translation_energy[i] for i in range(0,n_steps-2)]
plot4 = [Vibrational_energy[i] for i in range(0,n_steps-2)]
plot5 = [Rotational_energy[i] for i in range(0,n_steps-2)]
temps = time[0:n_steps-2]
plt.figure(figsize=(10,6))
plt.plot(temps, plot1, c="red", label="energie totale")
plt.plot(temps, plot2, c="blue", label="energie potentielle")
plt.plot(temps, plot3, c="green", label="energie de translation")
plt.plot(temps, plot4, c="orange", label="energie de vibration")
plt.plot(temps, plot5, c="yellow", label="energie rotative")
plt.legend()
plt.grid(True)
plt.show()

# energie_potentiel = [0.5*k*(distance(H_positions[i], Cl_positions[i])-r_eq)**2 for i in range(0,70000)]
# mu = (m_H*m_Cl)/(m_Cl+m_H)
# energie_rel = [0.5*mu*((Cl_velocity[i][0]-H_velocity[i][0])**2+(Cl_velocity[i][1]-H_velocity[i][1])**2+(Cl_velocity[i][2]-H_velocity[i][2])**2) for i in range(0,70000)]
# energie_CM = [0.5*(m_H+m_Cl)*((m_H*H_velocity[i][0]+m_Cl*Cl_velocity[i][0])/(m_Cl+m_H))**2 for i in range(0,70000)]
# temps = time[0:70000]
# energie_tot = [energie_potentiel[i]+energie_rel[i] for i in range(0,70000)]
# plt.figure(figsize=(10,6))
# plt.plot(temps, energie_tot)
# plt.plot(temps, energie_rel)
# plt.plot(temps, energie_potentiel)
# # plt.plot(temps, energie_CM)
# plt.xlabel('Temps (s)')
# plt.grid(True)
# plt.show()

#Visualisation de la distance entre H et Cl au cours du temps
# distances = [distance(H_positions[i], Cl_positions[i]) for i in range(0,n_steps-2)]
# temps = time[0:n_steps-2]
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