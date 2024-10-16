import numpy as np

def solve_verlet(R0, F, dt, N, Vi=None):
    if Vi is None:
        Vi = np.zeros(R0.shape())
    R = [R0]
    R1 = R0 + Vi*dt (dt**2)*F(R0)
    R.append(R1)

    for k in range(N-1):
        Rk = 2*R[-1] - R[-2] + F(R[-1])*(dt**2)
        R.append(Rk)

    return R


