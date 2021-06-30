#!/usr/bin/python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def dlqr(A, B, Q, R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    print(P.shape)
    print(scipy.linalg.inv(B.T @ P @ B + R).shape)
    K = np.array(scipy.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A))
    return K


dt = 0.2  # 20 ms
A = np.array([
    [1.0, 0.0, dt, 0.0],
    [0.0, 1.0, 0.0, dt],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
B = np.array([
    [0.5 * dt ** 2, 0.0],
    [0.0, 0.5 * dt ** 2],
    [dt, 0.0],
    [0.0, dt]
])
print(A)
print(B)

Q = np.eye(4)
R = np.eye(2) * 0.1

K = dlqr(A, B, Q, R)
print(K)

nsteps = 250
time = np.linspace(0, 2, nsteps, endpoint=True)
xk = np.array([[3.0],
               [5.0],
               [0.0],
               [0.0]])
xref = np.array([[7.0],
                 [2.0],
                 [0.0],
                 [0.0]])

x = np.zeros((nsteps, 4))

for i, t in enumerate(time):
    uk = -K @ (xk - xref)
    xk = A @ xk + B @ uk
    x[i] = xk.flatten()

x = np.vstack(x)
plt.plot(time, x[:, 0], label="cart position, meters")
plt.plot(time, x[:, 1], label="cart position, meters")
plt.legend(loc='upper right')
plt.grid()
plt.show()
