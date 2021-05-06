import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math


class DroneDynamics(object):
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt
        self.state_dim = 7
        self.ctrl_dim = 4

        # Discrete Dynamics
        self.A = np.eye(self.state_dim)
        self.A[0:3, 3:6] = np.eye(3) * self.dt
        self.B = np.zeros((self.state_dim, self.ctrl_dim))
        self.B[:3, :3] = 0.5 * np.eye(3) * self.dt ** 2
        self.B[3:, :] = np.eye(4) * self.dt

        # Quadratic Costs
        self.Q = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 0.01])
        self.R = np.eye(self.ctrl_dim) * 0.1

        S = np.array(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        K = np.array(scipy.linalg.inv(self.B.T @ S @ self.B + self.R) @ (self.B.T @ S @ self.A))

        # Lifted Dynamics for fast LQR rollout
        # xtraj = Abar @ x0 + Bbar @ [xg, xg, ... xg]
        self.Abar = self.build_a_bar(self.A, self.B, K)
        self.Bbar = self.build_b_bar(self.A, self.B, K)

        S = [None for i in range(self.N + 1)]  # P
        self.Ks = [None for i in range(self.N)]
        S[-1] = 10 * self.Q
        for k in range(self.N - 1, -1, -1):
            self.Ks[k] = scipy.linalg.inv(self.R + self.B.T @ S[k + 1] @ self.B) @ self.B.T @ S[k + 1] @ self.A
            S[k] = self.Q + self.A.T @ S[k + 1] @ self.A - self.A.T @ S[k + 1] @ self.B @ self.Ks[k]

    def rollout_with_time(self, x0, xg):
        t0 = x0[-1]
        ts = t0 + np.arange(0, self.N) * self.dt
        xs = self.rollout(x0[:-1], xg[:-1])
        return np.hstack([xs, ts[:, np.newaxis]])

    def rollout(self, x0, xg):
        """
        solve a discrete Algebraic Riccati equation (DARE)
        """
        # x = np.copy(x0)
        # xs = np.zeros((self.N, self.state_dim))
        # for i in range(self.N):
        #     x = self.A @ x + self.B @ -self.Ks[i] @ (x - xg)
        #     xs[i, :] = x
        #
        # return xs
        xs = self.Abar @ x0 + self.Bbar @ np.tile(xg, (self.N))
        return xs.reshape(self.N, self.state_dim)

    def build_a_bar(self, A, B, K):
        rm = cm = A.shape[0]
        A_bar = np.zeros((rm * (self.N), cm))
        for i in range(self.N):
            A_bar[rm * i:rm * (i + 1), :] = np.linalg.matrix_power(A - B @ K, i + 1)
        return A_bar

    def build_b_bar(self, A, B, K):
        rm = cm = A.shape[0]
        B_bar = np.zeros((rm * (self.N), cm * (self.N)))
        for r in range(self.N):
            for c in range(self.N):
                order = r - c
                if order < 0:
                    B_bar[rm * r:rm * (r + 1), cm * c:cm * (c + 1)] = np.zeros_like(A)
                else:
                    B_bar[rm * r:rm * (r + 1), cm * c:cm * (c + 1)] = (
                            np.linalg.matrix_power(A - B @ K, order) @ B @ K)
        return B_bar
