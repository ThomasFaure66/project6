import numpy as np
import matplotlib.pyplot as plt

from potentials import distance, force_harmonique, force_morse, k, r_eq, pot_morse, pot_har
from solver import solve_verlet, solve_verlet_lang

# Constantes physiques

r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)
m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)
kB = 1.380e-23 # Constante de Boltzmann
T_ther = 300 # Température thermostat
gamma = 1e12
# Paramètres de simulation

dt = 1e-16 # Pas de temps (s)
n_steps = 100000  # Nombre de pas de temps

# Lancement de la simulation

H_pos0 = np.array([r_eq, 0.0, 0.0])  # Position initiale de H
Cl_pos0 = np.array([0.0, 0.0, 0.0])  # Position initiale de Cl (à r_eq de H)

H_positions, Cl_positions, H_velocity, Cl_velocity = solve_verlet_lang(H_pos0=H_pos0,
                                                                        Cl_pos0=Cl_pos0,
                                                                        F = force_harmonique,
                                                                        N = n_steps,
                                                                        dt=dt,
                                                                        gamma = gamma,
                                                                        T=10
)
#time, H_positions, Cl_positions, H_velocity, Cl_velocity = verlet_3d()

# Conversion des résultats pour la visualisation
H_positions = np.array(H_positions)
Cl_positions = np.array(Cl_positions)
H_velocity = np.array(H_velocity)
Cl_velocity = np.array(Cl_velocity)

print(H_positions.shape)
print(Cl_positions.shape)

### Clacul des observables

mu = (m_H*m_Cl)/(m_Cl+m_H)
Mtot = (m_H+m_Cl)

CM_velocity = (m_H*H_velocity+m_Cl*Cl_velocity)/Mtot
Relative_position = Cl_positions - H_positions
Relative_velocity = Cl_velocity - H_velocity
r_norm = np.linalg.norm(Relative_position, axis=1)
r_dot2 = np.einsum("ij,ij->i",Relative_position, Relative_velocity)/r_norm
omega = np.linalg.norm(np.cross(Relative_position, Relative_velocity), axis=1) / (r_norm**2)
I = mu*(r_norm**2)

Translation_energy = 0.5 * (m_Cl+m_H)*np.linalg.norm(CM_velocity, axis = 1)**2
Vibrational_energy = 0.5*(mu)*r_dot2**2
Rotational_energy = 0.5*I*omega**2
Potential_energy = pot_har(r_norm-r_eq)

Kinetic_energy = Translation_energy + Vibrational_energy + Rotational_energy
Total_energy = Potential_energy + Kinetic_energy

### Affichage

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