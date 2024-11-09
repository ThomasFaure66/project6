import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from math import exp

D = 4.6141
alpha = 1.81e10
r_eq = 1.27e-10
T_ther = 10000
kB = 1.380e-23

print(3.5*kB*T_ther/(1.6e-19))
def Morse(r):
    return(D*(np.exp(-alpha*(r-r_eq))-1)**2)

def Harmonique(r):
    return D*(alpha**2)*(r-r_eq)**2

x_values = np.linspace(0.7e-10, 2.5e-10, 100)
y_values = Morse(x_values)
z_values = Harmonique(x_values)


plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=r"$Morse$", color="blue")
plt.plot(x_values, z_values, label =r"$Harmonique$", color="red")
plt.axhline(3.5*kB*T_ther/(1.6e-19), color='green', linestyle='--', label=r"Energie Totale (eV)")

plt.xlabel("r")
plt.ylabel("Potentiel")
plt.title("Potentiel")
plt.legend()
plt.grid(True)
plt.show()