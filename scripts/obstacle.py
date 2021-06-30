import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math


class ObstacleBicycle(object):
    def __init__(self, T, r, dt=0.1, vmax=1, max_steer_psi=0.05, max_steer_phi=0.05, num_angles=10, traj=None):
        self.state_dim = 5  # x, y, z, theta, phi
        self.space_dim = 3  # x, y, z
        self.r = r
        self.traj = traj
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.stationary_tol = 0.1
        self.vmax = vmax

        # reachable set
        steer_angles_psi = np.linspace(-max_steer_psi, max_steer_psi, num_angles)
        steer_angles_phi = np.linspace(-max_steer_phi, max_steer_phi, num_angles)
        self.reachable_trajs = []
        for steer_psi in steer_angles_psi:
            for steer_phi in steer_angles_phi:
                action = (vmax, (steer_psi, steer_phi))
                traj = self.rollout_const_vel(action)
                self.reachable_trajs.append(traj[np.newaxis, :])
        self.reachable_trajs = np.vstack(self.reachable_trajs)

    def calc_reachable_endpoints(self, pos, yaw, pitch, velocity):
        if velocity < self.stationary_tol:
            return None

        # scale predicted reachable set down if velocity < vmax
        # this assumes vmax reachable set is a superset of any lower velocity's set
        # velocity = min(self.vmax, velocity)
        # idx = int(np.rint((self.N - 1) * velocity / self.vmax))
        idx = -1
        # apply translation and rotation
        endpoints = self.reachable_trajs[:, idx, :self.space_dim]  # don't include theta in positions
        rot = Rotation.from_euler("XYZ", [pitch, 0, yaw]).as_matrix()
        endpoints = (rot @ endpoints.T).T + pos[np.newaxis, :]
        return endpoints

    def rollout_const_vel(self, action, x0=None):
        v, (steerxy, steerz) = action

        traj = np.zeros(shape=(self.N, self.state_dim))
        if x0 is None: x0 = np.zeros(self.state_dim)
        x = np.copy(x0)
        for i in range(self.N):
            theta, phi = x[-2:]

            # state derivatives
            thetadot = v * math.tan(steerxy)
            phidot = v * math.tan(steerz)

            vxy = v * math.cos(phi)
            vz = v * math.sin(phi)
            xdot = vxy * math.cos(theta)
            ydot = vxy * math.sin(theta)
            zdot = vz
            state_dot = np.array([xdot, ydot, zdot, thetadot, phidot])

            # update state and store
            x = x + state_dot * self.dt
            traj[i] = x

        return traj

    def bounding_box_from_points(self, points):
        # min and max across all points
        min_state = np.min(points, axis=0)
        max_state = np.max(points, axis=0)
        return min_state, max_state

    def update_traj(self, traj):
        if not isinstance(traj, np.ndarray):
            traj = np.vstack(traj)
        self.traj = traj

    def is_intersect(self, other_pos, r, t, debug=False):
        # (dist < r1 + r2)**2
        # don't predict motion for t > T outisde predicted trajectory
        # mathematically equivalent to assigning equal probability to all of reachable set
        # with finite discretization and over a large enough area, probabilities are just 0
        ti = min(int(t / self.dt), self.N - 1)
        obs_pos = self.traj[ti, :self.space_dim]

        if ti > 0:
            vec = self.traj[ti, :self.space_dim] - self.traj[ti - 1, :self.space_dim]
        else:
            vec = self.traj[ti + 1, :self.space_dim] - self.traj[ti, :self.space_dim]

        velocity = np.linalg.norm(vec)

        yaw = math.atan2(vec[1], vec[0])
        pitch = math.atan2(vec[2], vec[0])

        endpoints = self.calc_reachable_endpoints(obs_pos, yaw, pitch, velocity)
        if endpoints is not None:
            # AABB collision check: https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
            min_corners, max_corners = self.bounding_box_from_points(np.vstack([endpoints, obs_pos[np.newaxis, :]]))
            minx, miny, minz = min_corners
            maxx, maxy, maxz = max_corners

            cx, cy, cz = other_pos

            # find closest point to sphere
            nearest_x = max(minx, min(cx, maxx))
            nearest_y = max(miny, min(cy, maxy))
            nearest_z = max(minz, min(cz, maxz))

        else:
            nearest_x, nearest_y, nearest_z = obs_pos

        sq_dist = np.sum((np.array([nearest_x, nearest_y, nearest_z]) - other_pos) ** 2)
        if debug:
            return (sq_dist < (self.r + r) ** 2,
                    [minx, maxx, miny, maxy, minz, maxz],
                    endpoints,
                    (nearest_x, nearest_y, nearest_z))

        else:
            return sq_dist < (self.r + r) ** 2
