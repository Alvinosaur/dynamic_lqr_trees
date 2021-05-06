import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math

from obstacle import ObstacleBicycle
from dynamics import DroneDynamics


class ClosedLoopRRT(object):
    def __init__(self, N, dt, radius, space_dim, dist_tol, ds=None):
        self.radius = radius
        if ds is None:
            ds = 2 * self.radius
        self.ds = ds  # sampling granularity as distance
        self.space_dim = space_dim
        self.dist_tol = dist_tol
        self.model = DroneDynamics(N, dt)
        self.trajectory = None

    def reuse_valid(self, x0, obstacles):
        if self.trajectory is None:
            return False, False

        # find nearest point in trajectory
        distances = np.linalg.norm(self.trajectory[:, :self.space_dim] - x0[np.newaxis, :self.space_dim], axis=1)
        nearest_idx = np.argmin(distances)

        # # solve for trajectory to this new target and check for collisions
        # interm_traj = self.model.rollout_with_time(x0=x0, xg=self.trajectory[nearest_idx, :])
        # valid_samples, _ = self.filter_samples(interm_traj, obstacles)
        # if len(valid_samples) == 0:
        #     print("Collide with nearest state!")
        #     return False
        #
        # # check if old trajectory reachable
        # last_valid_idx = valid_samples[-1][1]
        # interm_distances = np.linalg.norm(
        #     interm_traj[:last_valid_idx, :self.space_dim] - x0[np.newaxis, :self.space_dim], axis=1)
        # close_idxs = np.where(interm_distances < self.dist_tol)[0]
        # if len(close_idxs) == 0:
        #     print("Doesn't reach nearest state!")
        #     return False

        # check if the rest of the trajectory is safe
        self.trajectory = self.trajectory[nearest_idx:]
        # # self.trajectory = np.vstack([interm_traj[:close_idxs[0]], ])  # update new trajectory
        self.trajectory[:, -1] -= self.trajectory[0, -1]  # update time so first state has 0 time
        _, has_collision = self.filter_samples(self.trajectory, obstacles)
        if has_collision:
            print("Rest of path has collision!")
            return False, False
        else:
            plan_changed = nearest_idx > 2
            return True, plan_changed

    def replan(self, x0, xg, obstacles, with_time=False):
        # Try reusing pre-existing plan if available
        # destructively modifies old waypoints and traj pieces
        is_valid, plan_changed = self.reuse_valid(x0, obstacles)
        if is_valid:
            print("Plan: %d" % plan_changed)
            return plan_changed

        state_dim = len(x0)
        found_path = False
        goali = None
        starti = 0
        nodes = [x0]
        parent = dict()

        count = 0
        while not found_path:
            count += 1
            # draw sample biased towards straight line
            new_x = self.sample_along_vec(x0, xg, N=1, D=self.space_dim).flatten()  # 1 x 2
            new_x = np.concatenate([new_x, np.zeros((state_dim - self.space_dim))])  # fill remaining states as 0

            # find nearest neighbor - Naive euclidean distance
            dists = np.linalg.norm(np.array(nodes)[:, :self.space_dim] - new_x[:self.space_dim], axis=1)
            nearesti = np.argmin(dists)

            # generate LQR-based trajectory filtered to avoid obstacles
            traj = self.model.rollout_with_time(x0=nodes[nearesti], xg=new_x)
            valid_samples, _ = self.filter_samples(traj, obstacles)

            # Debug
            # stepsize = max(1, int(len(traj) / num_steps))
            # valid_length = len(valid_samples)
            # all_trajs.append(traj[:valid_length * stepsize])

            for (s, traj_idx) in valid_samples:
                si = len(nodes)
                nodes.append(s)
                parent[si] = (nearesti, traj[:traj_idx + 1])

                # try connecting to goal too
                traj_to_goal = self.model.rollout_with_time(s, xg)
                valid_samples_goal, _ = self.filter_samples(traj_to_goal, obstacles)

                # Debug
                # stepsize = max(1, int(len(traj_to_goal) / num_steps))
                # valid_length  = len(valid_samples_goal)
                # goal_trajs.append(traj_to_goal[:valid_length * stepsize])

                for (s_goal, traj_to_goal_idx) in valid_samples_goal:
                    si_goal = len(nodes)
                    nodes.append(s_goal)
                    parent[si_goal] = (si, traj_to_goal[:traj_to_goal_idx + 1])

                    # check if close enough to goal
                    dist = np.linalg.norm(s_goal[:self.space_dim] - xg[:self.space_dim])
                    if dist < self.dist_tol:
                        goali = si_goal
                        found_path = True
                        break

        # Build full path from start to goal if exists
        if found_path:
            traj_pieces = []
            curi = goali
            while curi != starti:
                state = nodes[curi]
                traj_to_state = parent[curi][1]
                traj_pieces.append(traj_to_state)
                curi = parent[curi][0]

            traj_pieces.reverse()
            self.trajectory = np.vstack(traj_pieces)

            plan_changed = True
            return plan_changed

        else:
            raise (Exception("Failed to find a plan???"))

    def filter_samples(self, traj, obstacles):
        distances = np.linalg.norm(
            traj[1:, :self.space_dim] - traj[:-1, :self.space_dim],
            axis=1)
        distances = np.cumsum(distances)

        has_collision = False
        valid_samples = []
        si = 0
        while si < len(traj) - 1:
            next_si = np.nonzero(distances > self.ds)[0]
            if len(next_si) == 0:
                si = len(traj) - 1
            else:
                si = next_si[0]
                distances -= distances[si]

            sample_valid = True
            for obs in obstacles:
                pos = traj[si, :self.space_dim]
                t = traj[si, -1]
                if obs.is_intersect(pos, self.radius, t):
                    sample_valid = False
                    has_collision = True
                    break

            if sample_valid:
                valid_samples.append((traj[si], si))
            else:
                break

        return valid_samples, has_collision

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        # https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    @staticmethod
    # https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/
    # Biasing sampling along vector from start to goal
    def sample_along_vec(x0, xg, N, D):
        vec = xg[:D] - x0[:D]
        vec /= np.linalg.norm(vec)

        mag_vec = 0.2 * np.linalg.norm(xg[:D] - x0[:D])
        mag_orth = 0.3 * mag_vec
        if D == 2:
            orth_vec = np.array([vec[1], -vec[0]])
            L = np.diag([mag_vec, mag_orth])
            V = np.hstack([
                vec[:, np.newaxis], orth_vec[:, np.newaxis]
            ])
        elif D == 3:
            # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
            # (0,c,−b),(−c,0,a) and (−b,a,0) for any nonzero vec <a,b,c>
            a, b, c = vec.flatten()

            # rotation matrix from vector to z-axis
            rot = ClosedLoopRRT.rotation_matrix_from_vectors(vec, np.array([0, 0, 1]))
            orth_vec1 = rot @ np.array([1, 0, 0])
            orth_vec2 = rot @ np.array([0, 1, 0])
            L = np.diag([mag_vec, mag_orth, mag_orth])
            V = np.hstack([
                vec[:, np.newaxis], orth_vec1[:, np.newaxis], orth_vec2[:, np.newaxis]
            ])
        else:
            raise (NotImplementedError("Only supports 2D or 3D"))

        mu = ((xg[:D] + x0[:D]) / 2).flatten()
        cov = V @ L @ V.T
        points = np.random.multivariate_normal(mu, cov, N)
        return points
