import numpy as np
import matplotlib.pyplot as plt

from potentials import distance, force_harmonique, force_morse, k, r_eq, pot_morse, pot_har
from solver import solve_verlet, solve_verlet_lang, moyenne_cumul

# Constantes physiques

r_eq = 1.27e-10  # Distance d'équilibre de la liaison HCl (m)
m_H = 1.00784 / (6.022e23)  # Masse de l'atome H (kg)
m_Cl = 35.453 / (6.022e23)  # Masse de l'atome Cl (kg)
kB = 1.380e-23 # Constante de Boltzmann
T_ther = 500 # Température thermostat
gamma = 1e14
# Paramètres de simulation

dt = 1e-16 # Pas de temps (s)
n_steps = 10000  # Nombre de pas de temps

# Lancement de la simulation

H_pos0 = np.array([r_eq, 0.0, 0.0])  # Position initiale de H
Cl_pos0 = np.array([-r_eq/20, 0.0, 0.0])  # Position initiale de Cl (à r_eq de H)
#temperatures = [10*k for k in range(10)]
#E, K, V = [], [], []

#for T in temperatures:
H_positions, Cl_positions, H_velocity, Cl_velocity = solve_verlet(H_pos0=H_pos0,
                                                                    Cl_pos0=Cl_pos0,
                                                                    F = force_harmonique,
                                                                    N = n_steps,
                                                                    dt=dt,
                                                                    H_vel0= np.sqrt(np.array([0.0, .5*kB*T_ther/m_H, 0.0]))
)
#time, H_positions, Cl_positions, H_velocity, Cl_velocity = verlet_3d()

# Conversion des résultats pour la visualisation
H_positions = np.array(H_positions)
Cl_positions = np.array(Cl_positions)
H_velocity = np.array(H_velocity)
Cl_velocity = np.array(Cl_velocity)

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

    # E.append(np.mean(Total_energy[100000:]/(kB*T)))
    # K.append(np.mean(Kinetic_energy[100000:]/(kB*T)))
    # V.append(np.mean(Potential_energy[100000:]/(kB*T)))

    # print(T)

# plt.scatter(temperatures, E, c = "red", label="Energie totale")
# plt.scatter(temperatures, K, c = "blue", label="Energie cinétique")
# plt.scatter(temperatures, V, c = "red", label="Energie Potentielle")
# plt.xlabel("Température (K)")
# plt.ylabel("E/kBT")
# plt.legend()
# plt.grid(True)
# plt.show()
### Affichage

# plot1 = [Total_energy[i] for i in range(0,n_steps)]
# plot2 = [Potential_energy[i] for i in range(0,n_steps)]
# plot3 = [Translation_energy[i] for i in range(0,n_steps)]
# plot4 = [Vibrational_energy[i] for i in range(0,n_steps)]
# plot5 = [Rotational_energy[i] for i in range(0,n_steps)]
# temps = [k*dt for k in range(n_steps)]
# plt.figure(figsize=(10,6))

plot1 = moyenne_cumul(Total_energy)
plot2 = moyenne_cumul(Rotational_energy)
plot3 = moyenne_cumul(Vibrational_energy)
plot4 = moyenne_cumul(Potential_energy)
temps = [k*1e-4 for k in range(n_steps+1)]
# plt.figure(figsize=(10,6))

plt.plot(temps, Total_energy/(kB*T_ther), c="red", label="Energie totale")
#plt.axhline(np.mean(Total_energy[-1000:]), color='red', linestyle='--', label='"energie totale moyenne')
plt.plot(temps, Rotational_energy/(kB*T_ther), c="blue", label="Energie de rotation")
#plt.axhline(np.mean(Potential_energy[-1000:]), color='blue', linestyle='--', label='"energie potentielle moyenne')
plt.plot(temps, Vibrational_energy/(kB*T_ther), c="green", label="Energie de vibration")
plt.plot(temps, Potential_energy/(kB*T_ther), c="orange", label="Energie Potentielle")
#plt.axhline(np.mean(Translation_energy[-1000:]), color='green', linestyle='--', label='"energie de translation moyenne')
#plt.plot(temps, plot4, c="orange", label="energie de vibration")
#plt.axhline(np.mean(Vibrational_energy[-1000:]), color='orange', linestyle='--', label='"energie de vibration moyenne')
#plt.plot(temps, plot5, c="yellow", label="energie rotative")
#plt.axhline(np.mean(Rotational_energy[-1000:]), color='yellow', linestyle='--', label='"energie rotationelle moyenne')
plt.ylabel('E/kbT')
plt.xlabel('Temps (ps)')
plt.legend()
plt.grid(True)
plt.savefig("equi_morse.png")
plt.show()

#Visualisation de la distance entre H et Cl au cours du temps
# distances = [distance(H_positions[i], Cl_positions[i]) for i in range(0,n_steps+1)]
# plt.figure(figsize=(10, 6))
# plt.plot(temps, distances)
# plt.axhline(r_eq, color='red', linestyle='--', label='Distance d\'équilibre')
# plt.xlabel('Temps (s)')
# plt.ylabel('Distance H-Cl (m)')
# plt.title('Distance entre H et Cl au cours du temps')
# plt.legend()
# plt.grid(True)
# plt.show() 