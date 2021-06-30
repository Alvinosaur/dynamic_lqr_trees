import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg


def solve_dare(A, B, Q, Qf, R, N):
    """
    solve a discrete Algebraic Riccati equation (DARE)
    """
    S = [None for i in range(N)]  # P
    K = [None for i in range(N - 1)]
    S[-1] = Qf
    for k in range(N - 2, -1, -1):
        K[k] = linalg.inv(R + B.T @ S[k + 1] @ B) @ B.T @ S[k + 1] @ A
        S[k] = Q + A.T @ S[k + 1] @ A - A.T @ S[k + 1] @ B @ K[k]

    return S, K


B = np.array([[0],
              [1.]])
Q = np.array([[1., 0.],
              [0., 1.]])
Qf = Q
R = np.array([[1.]])
N = 100
x = np.array([[3, 0.0]]).T
xref = np.array([[1.0, 0]]).T

xs = np.zeros((N + 1, 2))
xs[0, :] = x.flatten()

cont = True  # is continuous
dt = 0.1
if cont:
    A = np.array([[0., 1.],
                  [0., 0.]])

    S = np.matrix(linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(linalg.inv(R) * B.T * S)

    for i in range(1, N + 1):
        u = -K @ (x - xref)
        xdot = A @ x + B @ u
        x += xdot * dt
        xs[i, :] = x.flatten()

else:
    A = np.array([[1., 1],
                  [0., 1.]])

    S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(linalg.inv(B.T * S * B + R) * (B.T * S * A))
    print(K)
    Ss, Ks = solve_dare(A, B, Q, Qf, R, N)
    # S, K = Ss[0], Ks[0]
    for i in range(N - 1):
        print(Ks[i])

    for i in range(1, N):
        u = -Ks[i - 1] @ (x - xref)
        x = A @ x + B @ u
        xs[i, :] = x.flatten()

time_history = list(range(N + 1))

plt.plot(time_history, xs[:, 0], "-b", label="x1")
plt.plot(time_history, xs[:, 1], "-g", label="x2")

xref0_h = [xref[0] for i in range(N + 1)]
xref1_h = [xref[1] for i in range(N + 1)]
plt.plot(time_history, xref0_h, "--b", label="target x1")
plt.plot(time_history, xref1_h, "--g", label="target x2")

plt.show()
