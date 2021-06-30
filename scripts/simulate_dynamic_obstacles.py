#!/usr/bin/env python
import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelState
from visualization_msgs.msg import MarkerArray, Marker
import time
import numpy as np
from scipy.interpolate import interp1d

import utils


class Spline3D(object):
    def __init__(self, xs, ys, zs):
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.t0 = time.time()

    def __call__(self, t):
        try:
            return self.xs(t), self.ys(t), self.zs(t)
        except:
            import ipdb
            ipdb.set_trace()


class SimDynamicObstacles:
    def __init__(self):
        self.dt = 0.1
        self.T_complete = 7  # seconds for each obstacle to complete one pass
        self.T_pred = 3  # output ground truth path duration in seconds
        self.min_pts = 10
        self.max_pts = 20
        self.t0 = time.time()
        self.reversed = False
        self.noise = 0
        self.obs_radii = [0.3, 0.6, 0.7, 0.5, 0.2, 0.2]

        self.starts = np.array([
            [2, -2, 4],
            [6, -1, 3],
            [4, 1.5, 5],
            [8, 0, 3],
            [8, -2, 1],
            [7, -2, 3],  # [7, -4, 3],
        ])

        self.goals = np.array([
            [4, 2, 2],
            [6, 1, 3],
            [2, -1.5, 2],
            [8, 0, 3],
            [6, 2, 5],  # [6, 2, 8],
            [9, 2, 3],  # [9, 4, 3],
        ])
        speeds = np.linalg.norm(self.goals - self.starts, axis=1) / self.T_complete
        print(speeds)
        assert (np.sum(speeds > 1) == 0)

        self.num_obs = len(self.starts)

        self.splines = [self.gen_random_traj(self.starts[i], self.goals[i]) for i in range(self.num_obs)]
        self.obstacle_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10 * self.num_obs)
        self.obstacle_marker_pub = rospy.Publisher('/dynamic_lqr_trees/obstacle_markers', MarkerArray, queue_size=10)
        self.obstacle_path_pubs = [
            rospy.Publisher(f'/dynamic_lqr_trees/obs_path_{i}', Path, queue_size=10) for i in range(self.num_obs)]

    def reset_t0(self, t0):
        self.t0 = t0
        self.reversed = not self.reversed

    def gen_random_traj(self, start, goal):
        # generate random points along vector from start to goal
        num_pts = np.random.randint(low=self.min_pts, high=self.max_pts)
        vec = (goal - start).reshape(1, -1)

        # random length intervals
        t = np.concatenate(([0], np.sort(np.random.random(num_pts - 1))))
        t /= t[-1]
        t_reversed = (1 - t)[::-1]
        wpts = start + vec * t.reshape((num_pts, 1))

        # add noise to wpts
        wpts += np.random.uniform(low=-1, high=1, size=(num_pts, 3)) * self.noise
        x_spline = interp1d(x=t * self.T_complete, y=wpts[:, 0], kind="cubic")
        y_spline = interp1d(x=t * self.T_complete, y=wpts[:, 1], kind="cubic")
        z_spline = interp1d(x=t * self.T_complete, y=wpts[:, 2], kind="cubic")
        x_spline_reversed = interp1d(x=t_reversed * self.T_complete, y=wpts[::-1, 0], kind="cubic")
        y_spline_reversed = interp1d(x=t_reversed * self.T_complete, y=wpts[::-1, 1], kind="cubic")
        z_spline_reversed = interp1d(x=t_reversed * self.T_complete, y=wpts[::-1, 2], kind="cubic")

        return Spline3D(x_spline, y_spline, z_spline), Spline3D(x_spline_reversed, y_spline_reversed, z_spline_reversed)

    def publish_obstacles(self, t):
        # index into tuple of (original, reversed)
        pair_idx = 0 if not self.reversed else 1

        # visualizing ground truth trajectory
        t_cur = t - self.t0
        t_final = t_cur + self.T_pred
        t_forward = np.arange(start=t_cur, stop=min(t_final, self.T_complete), step=self.dt)
        if t_final > self.T_complete:
            t_rev = np.arange(start=self.T_complete, stop=t_final, step=self.dt) - self.T_complete
        else:
            t_rev = None

        marker_array_msg = MarkerArray()

        for i in range(self.num_obs):
            x, y, z = self.splines[i][pair_idx](t_cur)
            new_pos = ModelState()
            new_pos.reference_frame = "world"
            new_pos.model_name = "obstacle%d" % i
            new_pos.pose.position.x = x
            new_pos.pose.position.y = y
            new_pos.pose.position.z = z
            self.obstacle_pose_pub.publish(new_pos)

            # visualize obstacle in Rviz
            marker = Marker()
            marker.header.frame_id = "world"
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = 0
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 2 * self.obs_radii[i]
            marker.scale.y = 2 * self.obs_radii[i]
            marker.scale.z = 2 * self.obs_radii[i]
            marker.ns = "obstacle"
            marker_array_msg.markers.append(marker)

            # publish ground truth path
            x_path, y_path, z_path = self.splines[i][pair_idx](t_forward)
            if t_rev is not None:
                rev_path = self.splines[i][1 - pair_idx](t_rev)
                x_path = np.concatenate([x_path, rev_path[0]])
                y_path = np.concatenate([y_path, rev_path[1]])
                z_path = np.concatenate([z_path, rev_path[2]])
            full_path = np.vstack([x_path, y_path, z_path]).T
            self.obstacle_path_pubs[i].publish(utils.create_path(traj=full_path, dt=self.dt))

        self.obstacle_marker_pub.publish(marker_array_msg)


if __name__ == '__main__':
    # seed for reproducibility
    np.random.seed(12345)

    rospy.init_node("sim_dynamic_obstacles")
    rate = rospy.Rate(hz=20)
    obstacle_sim = SimDynamicObstacles()
    while not rospy.is_shutdown():
        t = time.time()
        if (t - obstacle_sim.t0) >= obstacle_sim.T_complete:
            obstacle_sim.reset_t0(t)

        obstacle_sim.publish_obstacles(t)

        try:  # prevent garbage in console output when thread is killed
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
