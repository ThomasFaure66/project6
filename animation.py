import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation
from solver import solve_verlet
from potentials import force_harmonique

T = 10000
r_eq = 1.27e-10
m_H = 1.00784 / (6.022e23)
kB = 1.380e-23
nstep = 100000

Hp, Clp, Hv, Clv = solve_verlet(np.array([r_eq, 0.0, 0.0]),
                                np.array([r_eq/10, 0.0, 0.0]),
                                force_harmonique,
                                1e-16, 
                                N = nstep,
                                H_vel0= np.sqrt(np.array([0.0, 5*kB*T/m_H, 0.0]))
                                )

b = 3
fig = plt.figure(figsize = (7,7))
ax = fig.add_subplot(111, projection='3d')  

mol1 = ax.scatter(np.array([[]]), np.array([[]]), np.array([[]]))
mol2 = ax.scatter(np.array([[]]), np.array([[]]), np.array([[]]))

ax.set_xlim([-b, b])
ax.set_ylim([-b, b])
ax.set_zlim([-b, b])

plt.xlabel("å")
plt.ylabel("å")

def animate(j):
    i = 300*j
    global mol2
    global mol1
    mol1.remove()
    mol2.remove()
    mol1 = ax.scatter(Hp[i][0]*1e10, Hp[i][1]*1e10, Hp[i][2]*1e10, c='blue', s=100)
    mol2 = ax.scatter(Clp[i][0]*1e10, Clp[i][1]*1e10, Clp[i][2]*1e10, c='green', s=100)
    fig.canvas.draw()

anim = FuncAnimation(fig, animate, frames=300, interval=20)
plt.show()
anim.save("../hcl3.gif")
plt.close()
