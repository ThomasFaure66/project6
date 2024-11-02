import numpy as np
import matplotlib.pyplot as plt

from potentials import distance, force_harmonique, force_morse, k, r_eq
from solver import solve_verlet

# Constantes physiques

r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)
m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)
kB = 1.380e-23 # Constante de Boltzmann
T_ther = 300 # Température thermostat
gamma = 0
# Paramètres de simulation

dt = 1e-16 # Pas de temps (s)
n_steps = 10000  # Nombre de pas de temps

# Algorithme de Verlet pour l'intégration des équations de mouvement en 3D
def verlet_3d():
    # Initialisation des position
    # s et des vitesses dans l'espace 3D
    H_pos = np.array([r_eq, 0.0, 0.0])  # Position initiale de H
    Cl_pos = np.array([0.0, 0.0, 0.0])  # Position initiale de Cl (à r_eq de H)
    
    H_vel = np.array([0.0, np.sqrt(3*kB*10/m_H), 0.0])  # Vitesse initiale de H
    Cl_vel = np.array([0.0, 0.0, 0.0])  # Vitesse initiale de Cl
    
    # Listes pour enregistrer les positions au cours du temps
    H_positions = [H_pos, H_pos + H_vel*dt]
    Cl_positions = [Cl_pos, Cl_pos + Cl_vel*dt]
    H_velocity = [H_vel, H_vel]
    Cl_velocity = [Cl_vel, Cl_vel]
    time_list = [0]

    #Calcul du deuxième pas : 

    F = force_harmonique(H_pos, Cl_pos)

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
        F = force_harmonique(H_pos, Cl_pos)
        
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

H_pos0 = np.array([r_eq, 0.0, 0.0])  # Position initiale de H
Cl_pos0 = np.array([-r_eq/100000, 0.0, 0.0])  # Position initiale de Cl (à r_eq de H)
H_vel0 = np.array([0.0, 0.0, 0.0])  # Vitesse initiale de H
C_vel0 = np.array([0.0, 0.0, 0.0])  # Vitesse initiale de Cl

H_positions, Cl_positions, H_velocity, Cl_velocity = solve_verlet(H_pos0=H_pos0,
                                                                        Cl_pos0=Cl_pos0,
                                                                        F = force_morse,
                                                                        N = n_steps,
                                                                        dt=dt,
                                                                        H_vel0=H_vel0,
                                                                        Cl_vel0=C_vel0
)
#time, H_positions, Cl_positions, H_velocity, Cl_velocity = verlet_3d()

# Conversion des résultats pour la visualisation
H_positions = np.array(H_positions)
Cl_positions = np.array(Cl_positions)
H_velocity = np.array(H_velocity)
Cl_velocity = np.array(Cl_velocity)

mu = (m_H*m_Cl)/(m_Cl+m_H)
Mtot = (m_H+m_Cl)

CM_velocity = (m_H*H_velocity+m_Cl*Cl_velocity)/Mtot
Relative_position = Cl_positions - H_positions
Relative_velocity = Cl_velocity - H_velocity
r_norm = np.linalg.norm(Relative_position, axis=1)
r_dot2 = np.einsum("ij,ij->i",Relative_position, Relative_velocity)/r_norm
omega = np.linalg.norm(np.cross(Relative_position, Relative_velocity), axis=1) / (r_norm**2)


Translation_energy = 0.5 * (m_Cl+m_H)*np.linalg.norm(CM_velocity, axis = 1)**2
Vibrational_energy = 0.5*(mu)*r_dot2**2
# print("Energie de vibration", Vibrational_energy)
I = mu*(r_norm**2)
Rotational_energy = 0.5*I*omega**2

Potential_energy = 0.5*k*(r_norm-r_eq)**2

Total_energy = Potential_energy + Translation_energy + Vibrational_energy + Rotational_energy


plot1 = [Total_energy[i] for i in range(0,n_steps)]
plot2 = [Potential_energy[i] for i in range(0,n_steps)]
plot3 = [Translation_energy[i] for i in range(0,n_steps)]
plot4 = [Vibrational_energy[i] for i in range(0,n_steps)]
plot5 = [Rotational_energy[i] for i in range(0,n_steps)]
temps = [k*dt for k in range(n_steps)]
plt.figure(figsize=(10,6))

plt.plot(temps, plot1, c="red", label="energie totale")
#plt.axhline(np.mean(Total_energy[-1000:]), color='red', linestyle='--', label='"energie totale moyenne')
plt.plot(temps, plot2, c="blue", label="energie potentielle")
#plt.axhline(np.mean(Potential_energy[-1000:]), color='blue', linestyle='--', label='"energie potentielle moyenne')
plt.plot(temps, plot3, c="green", label="energie de translation")
#plt.axhline(np.mean(Translation_energy[-1000:]), color='green', linestyle='--', label='"energie de translation moyenne')
plt.plot(temps, plot4, c="orange", label="energie de vibration")
#plt.axhline(np.mean(Vibrational_energy[-1000:]), color='orange', linestyle='--', label='"energie de vibration moyenne')
plt.plot(temps, plot5, c="yellow", label="energie rotative")
#plt.axhline(np.mean(Rotational_energy[-1000:]), color='yellow', linestyle='--', label='"energie rotationelle moyenne')
plt.legend()
plt.grid(True)
plt.show()

#Visualisation de la distance entre H et Cl au cours du temps
distances = [distance(H_positions[i], Cl_positions[i]) for i in range(0,n_steps)]
plt.figure(figsize=(10, 6))
plt.plot(temps, distances)
plt.axhline(r_eq, color='red', linestyle='--', label='Distance d\'équilibre')
plt.xlabel('Temps (s)')
plt.ylabel('Distance H-Cl (m)')
plt.title('Distance entre H et Cl au cours du temps')
plt.legend()
plt.grid(True)
plt.show()