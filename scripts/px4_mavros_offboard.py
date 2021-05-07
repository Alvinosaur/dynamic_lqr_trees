#!/usr/bin/env python
import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget, AttitudeTarget, Thrust
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, Float64, String, Header

import time
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import math
import numpy as np

from inverseDyn import inverse_dyn
from drone_mpc.drone_mpc import DroneMPC
from closed_loop_rrt import ClosedLoopRRT
from obstacle import ObstacleBicycle
import utils
import threading


class Px4Controller:
    def __init__(self):
        # Discrete Dynamics
        self.N = 10
        self.dt = 0.1
        self.state_dim = 7
        self.ctrl_dim = 4
        self.A = np.eye(self.state_dim)
        self.A[0:3, 3:6] = np.eye(3) * self.dt
        self.B = np.zeros((self.state_dim, self.ctrl_dim))
        self.B[:3, :3] = 0.5 * np.eye(3) * self.dt ** 2
        self.B[3:, :] = np.eye(4) * self.dt

        self.Ac = np.eye(self.state_dim)
        self.Ac[0:3, 3:6] = np.eye(3)
        self.Bc = np.zeros((self.state_dim, self.ctrl_dim))
        self.Bc[:3, :3] = 0.5 * np.eye(3)
        self.Bc[3:, :] = np.eye(4)

        # Quadratic Costs
        self.Q = np.diag([20, 20, 20, 0.1, 0.1, 0.1, 0.1])
        self.R = np.diag([.1, .1, .1, 10])

        # MPC
        self.N_mpc = 10
        self.S = 5 * self.Q
        # self.gamma = np.linspace(start=1, stop=0.3, num=self.N_mpc + 1).reshape((self.N_mpc + 1, 1, 1))
        # self.Qs = np.tile(self.Q[np.newaxis, :], (self.N_mpc + 1, 1, 1)) * self.gamma
        # self.S = self.Qs[-1]

        # State and Control Constraints
        # x_constraints = np.array([[np.Inf, np.Inf, np.Inf, 2.0, 2.0, 2.0, np.Inf]]).T
        # x_constraints = np.hstack([-x_constraints, x_constraints])
        # u_constraints = np.array([[1.0, 1.0, 1.0, (math.pi / 16) * self.dt]]).T
        # u_constraints = np.hstack([-u_constraints, u_constraints])
        x_constraints = None
        u_constraints = None

        # motion planner
        self.radius = 0.5
        self.threshold_timeout = 0.5
        self.return_home_timeout = 3
        self.rate = rospy.Rate(int(1 / self.dt))  # Hz
        self.T = self.N * self.dt
        self.planner = ClosedLoopRRT(
            A=self.A, B=self.B, Q=self.Q, R=self.R, S=self.S,
            N=self.N, dt=self.dt, space_dim=3, dist_tol=0.1, ds=1.0, radius=self.radius)
        self.controller = DroneMPC(A=self.Ac, B=self.Bc, Q=self.Q, S=self.S, R=self.R,
                                   N=self.N_mpc, dt=self.dt,
                                   x_constraints=x_constraints, u_constraints=u_constraints)
        self.t_offset = 0
        self.t_solved = time.time() + self.t_offset
        self.solver_thread = None
        self.Gff = np.array([0, 0, 9.8, 0])
        self.time_to_solve = 3
        self.path_index = 0

        # Takeoff
        self.x0 = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.xg = np.array([10.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.yaw_index = len(self.x0) - 2  # 2nd to last value

        self.padding = 0.2
        self.obs_radii = np.array([0.3, 0.6, 0.7, 0.5, 0.2, 0.2]) + self.padding
        self.dist_threshes = self.obs_radii + self.radius - self.padding  # distance = sum of two objects' radii
        self.num_obs = len(self.obs_radii)
        self.obstacles = [
            ObstacleBicycle(T=self.T, r=self.obs_radii[i], vmax=1,
                            max_steer_phi=0.5, max_steer_psi=0.5, num_angles=2) for
            i in range(self.num_obs)]

        self.local_pose = None
        self.local_pos = None
        self.local_q = None
        self.local_vel = None

        self.mavros_state = None
        self.arm_state = False
        self.flight_mode = "Land"

        self.X_mpc = None
        self.U_mpc = None
        self.solved = False

        # Analysis of Performance
        self.collision_count = 0

        # Attitude message
        self.att = AttitudeTarget()
        self.att.body_rate = Vector3()
        self.att.header = Header()
        self.att.header.frame_id = "base_footprint"

        # Subscribers
        self.local_vel_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped,
                                              self.local_vel_callback,
                                              queue_size=1)
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback,
                                               queue_size=1)
        self.mavros_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state_callback, queue_size=1)
        self.obstacle_path_subs = [
            rospy.Subscriber(f'/dynamic_lqr_trees/obs_path_{i}', Path, self.obstacle_path_cb, i) for i in
            range(self.num_obs)]

        # Publishers
        self.pos_control_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.att_control_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        # self.att_control_pub = rospy.Publisher('mavros/setpoint_attitude/attitude', PoseStamped, queue_size=10)
        self.thrust_control_pub = rospy.Publisher('mavros/setpoint_attitude/thrust', Thrust, queue_size=10)
        self.output_path_pub = rospy.Publisher('/hawkeye/drone_path', Path, queue_size=10)
        self.output_path_mpc_pub = rospy.Publisher('/hawkeye/drone_path_mpc', Path, queue_size=10)

        # Services
        self.armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)

    @staticmethod
    def quat_from_pose(pose_msg):
        return np.array([pose_msg.pose.orientation.x,
                         pose_msg.pose.orientation.y,
                         pose_msg.pose.orientation.z,
                         pose_msg.pose.orientation.w])

    def check_connection(self):
        for i in range(10):
            if self.local_pose is not None and self.local_vel is not None:
                break
            else:
                print("Waiting for initialization.")
                time.sleep(0.5)

        return self.local_pose is not None and self.local_vel is not None

    def reset_to_position(self, pos):
        max_count = 100  # 200 * 0.1 sec = 10 sec for takeoff
        pos_msg = self.construct_pose_target(*pos)
        reached_target = False
        count = 0
        while not reached_target and count < max_count:
            self.pos_control_pub.publish(pos_msg)
            self.arm_state = self.arm()
            self.offboard_state = self.offboard()
            time.sleep(0.2)
            count += 1
            reached_target = self.reached_target(self.local_pos, pos, threshold=0.5)

        return reached_target

    def takeoff(self):
        # fly straight down to the ground
        ground_pos = np.copy(self.local_pos)
        ground_pos[-1] = 1
        if not self.reset_to_position(ground_pos):
            return False

        # reset to origin
        home_pos_hover = np.array([0, 0, 1.0])
        if not self.reset_to_position(home_pos_hover):
            return False

        # takeoff to x0
        if not self.reset_to_position(self.x0[:3]):
            return False

        return True

    def control_attitude(self, body_rate=None, target_q=None, thrust=0.705):
        if body_rate is not None:
            self.att.body_rate.x = body_rate[0]
            self.att.body_rate.y = body_rate[1]
            self.att.body_rate.z = body_rate[2]
        else:
            assert (target_q is not None)
            self.att.orientation = Quaternion(*target_q)
            self.att.type_mask = 7  # ignore body rate
        self.att.header.stamp = rospy.Time.now()
        self.att.thrust = thrust
        self.att_control_pub.publish(self.att)

    def construct_pose_target(self, x, y, z, q=np.array([0, 0, 0, 1])):
        target_raw_pose = PoseStamped()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.pose.position.x = x
        target_raw_pose.pose.position.y = y
        target_raw_pose.pose.position.z = z

        target_raw_pose.pose.orientation.x = q[0]
        target_raw_pose.pose.orientation.y = q[1]
        target_raw_pose.pose.orientation.z = q[2]
        target_raw_pose.pose.orientation.w = q[3]

        return target_raw_pose

    def generate_action(self):
        if self.local_pose is None or self.local_vel is None:
            return None

        q = self.local_q
        roll, pitch, yaw = Rotation.from_quat(q).as_euler("XYZ")

        x = np.array([
            self.local_pose.pose.position.x,
            self.local_pose.pose.position.y,
            self.local_pose.pose.position.z,
            self.local_vel.twist.linear.x,
            self.local_vel.twist.linear.y,
            self.local_vel.twist.linear.z,
            yaw,
            0.0  # reset time to 0, assumes obstacles published fast enough
        ])

        start_time = time.time()
        plan_changed, nearest_idx = self.planner.replan(
            x, self.xg, obstacles=self.obstacles, replan=self.solved)

        self.path_index = nearest_idx + 1

        print("Didn't change")

        # execute next step
        # TODO: this might need to be rollout_with_time if waypoints
        # self.X_mpc = self.planner.model.rollout(x[:-1], self.waypoints[1])

        end_time = time.time()
        self.time_to_solve = end_time - start_time

        print("Finished solving in time %.3f" % (end_time - start_time))
        self.t_solved = time.time()
        self.solved = True

    def use_prev_traj(self):
        # try:
        self.planner.lock.acquire()
        start = min(self.path_index, len(self.planner.trajectory) - 1)
        end = min(start + self.N_mpc + 1, len(self.planner.trajectory))
        remainder = self.N_mpc + 1 - (end - start)
        xref = self.planner.trajectory[start:end, :self.yaw_index + 1]

        # duplicate last state for remainder of trajectory
        xref = np.vstack([xref, np.tile(xref[-1, :], (remainder, 1))])
        self.planner.lock.release()
        print(np.array2string(xref[:, :3], precision=2))

        # xref, yref, zref = self.planner.trajectory[start, :3]
        # yaw_ref = self.planner.trajectory[start, self.yaw_index] % (2 * math.pi)
        # except IndexError:
        #     # This is a race condition where thread finishes and changes planner.trajectory after we measure its length
        #     # update path index again
        #     start = min(self.path_index, len(self.planner.trajectory) - 1)
        #     end = min(start + self.N, len(self.planner.trajectory))
        #     remainder = self.N - (end - start)
        #     xref = self.planner.trajectory[start:end, :self.state_dim + 1]
        #     xref = np.vstack()
        #
        #     # xref, yref, zref = self.planner.trajectory[start, :3]
        #     # yaw_ref = self.planner.trajectory[start, self.yaw_index] % (2 * math.pi)

        yaw = Rotation.from_quat(self.local_q).as_euler("XYZ")[-1]
        x = np.array([[
            self.local_pose.pose.position.x,
            self.local_pose.pose.position.y,
            self.local_pose.pose.position.z,
            self.local_vel.twist.linear.x,
            self.local_vel.twist.linear.y,
            self.local_vel.twist.linear.z,
            yaw
        ]]).T  # state_dim x 1
        U_mpc, X_mpc = self.controller.solve(x, xref.T)
        self.output_path_mpc_pub.publish(
            utils.create_path(traj=X_mpc, dt=self.dt, frame="world"))

        # u = U_mpc[0] + self.Gff
        # thrust, phi, theta, psi = inverse_dyn(q=self.local_q, x_ref=x[0], u=u)

        x = X_mpc[1]
        target_q = Rotation.from_euler("XYZ", [0, 0, x[self.yaw_index]]).as_quat()
        pose_msg = self.construct_pose_target(x=x[0], y=x[1], z=x[2], q=target_q)

        return pose_msg

    def reached_target(self, cur_p, target_p, threshold=0.1):
        return np.linalg.norm(cur_p - target_p) < threshold

    def obstacle_path_cb(self, msg, i):
        traj = [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses]
        self.obstacles[i].update_traj(traj)

    def local_pose_callback(self, msg):
        self.local_pose = msg
        self.local_q = self.quat_from_pose(msg)
        self.local_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def local_vel_callback(self, msg):
        self.local_vel = msg

    def mavros_state_callback(self, msg):
        self.mavros_state = msg.mode

    def arm(self):
        if self.armService(True):
            return True
        else:
            print("Vehicle arming failed!")
            return False

    def disarm(self):
        if self.armService(False):
            return True
        else:
            print("Vehicle disarming failed!")
            return False

    def offboard(self):
        # DO NOT USE, it's safer to manually switch on offboard mode
        if self.flightModeService(base_mode=220, custom_mode=State.MODE_PX4_OFFBOARD):
            return True
        else:
            print("Vehicle Offboard failed")
            return False

    def check_collision(self):
        obs_poses = np.array([
            self.obstacles[i].traj[0] for i in range(self.num_obs)
        ])
        distances = np.linalg.norm(obs_poses - self.local_pos[np.newaxis, :], axis=1)
        return np.sum(distances < self.dist_threshes) > 0

    def plan_execute(self):
        reached_target = False
        while not reached_target:
            if self.check_collision():
                self.collision_count += 1
                return

            if self.reached_target(self.local_pos, target_p=self.xg[:3], threshold=0.3):
                print("Reached Target!")
                return

            if self.mavros_state == State.MODE_PX4_OFFBOARD:
                if self.solver_thread is None or not self.solver_thread.is_alive():
                    # Spawn a process to run this independently:
                    self.solver_thread = threading.Thread(target=self.generate_action)
                    self.solver_thread.start()

                if not self.solved:
                    # print("Staying at origin!")
                    desired_pos = self.construct_pose_target(x=self.x0[0], y=self.x0[1], z=self.x0[2])
                    self.pos_control_pub.publish(desired_pos)
                else:
                    self.pos_control_pub.publish(self.use_prev_traj())

                if self.planner.trajectory is not None:
                    self.output_path_pub.publish(
                        utils.create_path(traj=self.planner.trajectory, dt=self.dt, frame="world"))

            try:  # prevent garbage in console output when thread is killed
                self.rate.sleep()
            except rospy.ROSInterruptException:
                return

    def start(self):
        if not self.check_connection():
            print("Failed to connect!")
            return

        iter = 0
        num_trials = 10
        while not rospy.is_shutdown() and iter < num_trials:
            iter += 1
            # takeoff to reach desired tracking height
            if not self.takeoff():
                print("Failed to takeoff!")
                return
            print("Successful Takeoff!")

            self.plan_execute()
            self.solved = False

            try:  # prevent garbage in console output when thread is killed
                self.rate.sleep()
            except rospy.ROSInterruptException:
                break

        print("Num Collisions: %d" % self.collision_count)
        print("Collision Rate: %.3f" % (self.collision_count / num_trials))


if __name__ == '__main__':
    rospy.init_node("offboard_node")
    con = Px4Controller()
    con.start()
