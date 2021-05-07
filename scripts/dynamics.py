import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math

from drone_mpc.drone_mpc import DroneMPC


class DroneDynamics(object):
    def __init__(self, A, B, Q, R, S, N, dt):
        self.N = N
        self.dt = dt
        self.state_dim = A.shape[0]
        self.ctrl_dim = B.shape[1]

        self.A = A
        self.B = B
        self.Q = Q
        self.S = S  # terminal cost weights
        self.R = R

        S = np.array(scipy.linalg.solve_discrete_are(self.A, self.B, self.S, self.R))
        K = np.array(scipy.linalg.inv(self.B.T @ S @ self.B + self.R) @ (self.B.T @ S @ self.A))

        # Lifted Dynamics for fast LQR rollout
        # xtraj = Abar @ x0 + Bbar @ [xg, xg, ... xg]
        self.Abar = self.build_a_bar(self.A, self.B, K)
        self.Bbar = self.build_b_bar(self.A, self.B, K)

        # x_constraints = np.array([[np.Inf, np.Inf, np.Inf, 2.0, 2.0, 2.0, np.Inf]]).T
        # x_constraints = np.hstack([-x_constraints, x_constraints])
        # u_constraints = np.array([[1.0, 1.0, 1.0, (math.pi / 16) * self.dt]]).T
        # u_constraints = np.hstack([-u_constraints, u_constraints])
        self.drone_mpc = DroneMPC(A=self.A, B=self.B, Q=self.Q, S=self.S, R=self.R,
                                  N=self.N, dt=self.dt,
                                  x_constraints=None, u_constraints=None)

        # S = [None for i in range(self.N + 1)]  # P
        # self.Ks = [None for i in range(self.N)]
        # S[-1] = 10 * self.Q
        # for k in range(self.N - 1, -1, -1):
        #     self.Ks[k] = scipy.linalg.inv(self.R + self.B.T @ S[k + 1] @ self.B) @ self.B.T @ S[k + 1] @ self.A
        #     S[k] = self.Q + self.A.T @ S[k + 1] @ self.A - self.A.T @ S[k + 1] @ self.B @ self.Ks[k]

    def rollout_with_time(self, x0, xg):
        t0 = x0[-1]
        ts = t0 + np.arange(0, self.N) * self.dt
        xs = self.rollout(x0[:-1], xg[:-1])
        return np.hstack([xs, ts[:, np.newaxis]])

    def rollout(self, x0, xg):
        """
        solve a discrete Algebraic Riccati equation (DARE)
        """
        x0 = x0[:, np.newaxis]
        xg = xg[np.newaxis]
        xref = np.vstack([xg, np.tile(xg, (self.N, 1))])
        U_mpc, X_mpc = self.drone_mpc.solve(x0, xref.T)
        return X_mpc[1:]

        # xs = self.Abar @ x0 + self.Bbar @ np.tile(xg, self.N)
        # return xs.reshape(self.N, self.state_dim)

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
