#!/usr/bin/env python
import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, Float64, String, Header
# import tf_conversions
# import tf2_ros

import time
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import math
import numpy as np
import threading
from simple_pid import PID
from scipy.interpolate import interp1d


class Spline3D(object):
    def __init__(self, xs, ys, zs):
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.t0 = time.time()

    def __call__(self, t):
        return self.xs(t), self.ys(t), self.zs(t)


class SimDynamicObstacles:
    def __init__(self):
        self.T_complete = 7  # seconds for each obstacle to complete one pass
        N = 3  # num obstacles
        self.N = N
        self.min_pts = 10
        self.max_pts = 20
        self.t0 = time.time()
        self.reversed = False
        self.noise = 0

        self.starts = np.array([
            [2, 4, 3],
            [4, 3, 4],
            [7, 6, 2]
        ])

        self.goals = np.array([
            [3, -3, 4],
            [2, -4, 2],
            [7, -5, 3]
        ])

        self.splines = [self.gen_random_traj(self.starts[i], self.goals[i]) for i in range(N)]

        self.obstacle_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10 * N)

        '''
        ros services
        '''
        self.armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)

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
        for i in range(self.N):
            x, y, z = self.splines[i][pair_idx](t - self.t0)
            new_pos = ModelState()
            new_pos.reference_frame = "world"
            new_pos.model_name = "obstacle%d" % i
            new_pos.pose.position.x = x
            new_pos.pose.position.y = y
            new_pos.pose.position.z = z
            self.obstacle_pose_pub.publish(new_pos)


if __name__ == '__main__':
    # seed for reproducibility
    np.random.seed(12345)

    rospy.init_node("sim_dynamic_obstacles")
    rate = rospy.Rate(hz=20)
    obstacle_sim = SimDynamicObstacles()

    t0 = time.time()
    while not rospy.is_shutdown():
        t = time.time()
        if (t - t0) >= obstacle_sim.T_complete:
            t0 = t
            obstacle_sim.reset_t0(t0)

        obstacle_sim.publish_obstacles(t)

        try:  # prevent garbage in console output when thread is killed
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
