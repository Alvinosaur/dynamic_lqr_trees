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

from closed_loop_rrt import ClosedLoopRRT
from obstacle import ObstacleBicycle
import utils
import threading


class Px4Controller:
    def __init__(self):
        # motion planner
        self.threshold_timeout = 0.5
        self.return_home_timeout = 3
        self.N = 10
        self.dt = 0.1
        self.T = self.N * self.dt
        # self.drone_mpc = DroneMPC(N=self.N, dt=self.dt, load_solver=True)
        self.radius = 0.5
        self.planner = ClosedLoopRRT(N=self.N, dt=self.dt, space_dim=3, dist_tol=0.1, ds=1.0, radius=self.radius)
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

        self.padding = 0.1
        self.obs_radii = np.array([0.3, 0.6, 0.7, 0.5]) + self.padding
        self.num_obs = len(self.obs_radii)
        self.obstacles = [
            ObstacleBicycle(T=self.T, r=self.obs_radii[i], vmax=1,
                            max_steer_phi=0.5, max_steer_psi=0.5, num_angles=2) for
            i in range(self.num_obs)]

        self.local_pose = None
        self.local_q = None
        self.local_vel = None

        self.mavros_state = None
        self.arm_state = False
        self.flight_mode = "Land"

        self.X_mpc = None
        self.U_mpc = None
        self.solved = False

        # Attitude message
        self.att = AttitudeTarget()
        self.att.body_rate = Vector3()
        self.att.header = Header()
        self.att.header.frame_id = "base_footprint"

        rospy.init_node("offboard_node")
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

    def takeoff(self):
        max_count = 100  # 200 * 0.2 sec = 20 sec for takeoff
        takeoff_pos = self.construct_pose_target(
            x=self.x0[0], y=self.x0[1], z=self.x0[2])
        count = 0
        while not self.takeoff_detection() and count < max_count:
            self.pos_control_pub.publish(takeoff_pos)
            self.arm_state = self.arm()
            self.offboard_state = self.offboard()
            time.sleep(0.2)
            print("Height: %.3f" % self.local_pose.pose.position.z)
            count += 1

        return count < max_count

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
        plan_changed = self.planner.replan(
            x, self.xg, obstacles=self.obstacles)
        if plan_changed:
            self.path_index = 0

        # execute next step
        # TODO: this might need to be rollout_with_time if waypoints
        # self.X_mpc = self.planner.model.rollout(x[:-1], self.waypoints[1])

        end_time = time.time()
        self.time_to_solve = end_time - start_time

        print("Finished solving in time %.3f" % (end_time - start_time))
        self.t_solved = time.time()
        self.solved = True

    def use_prev_traj(self):
        try:
            path_index = min(self.path_index, len(self.planner.trajectory)-1)
            x, y, z = self.planner.trajectory[path_index, :3]
            yaw = self.planner.trajectory[path_index, self.yaw_index] % (2 * math.pi)
        except IndexError:
            # This is a race condition where thread finishes and changes planner.trajectory after we measure its length
            # update path index again
            path_index = min(self.path_index, len(self.planner.trajectory)-1)
            x, y, z = self.planner.trajectory[path_index, :3]
            yaw = self.planner.trajectory[path_index, self.yaw_index] % (2 * math.pi)

        print(path_index)
        # thrust, phi, theta, psi = self.drone_mpc.inverse_dyn(q=self.local_q, x_ref=self.X_mpc[1], u=self.U_mpc[path_index])

        target_q = Rotation.from_euler("XYZ", [0, 0, yaw]).as_quat()
        pose_msg = self.construct_pose_target(x=x, y=y, z=z, q=target_q)

        return pose_msg

    def reached_target(self, cur_p, target_p, threshold=0.1):
        return np.linalg.norm(cur_p - target_p) < threshold

    def obstacle_path_cb(self, msg, i):
        traj = [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses]
        self.obstacles[i].update_traj(traj)

    def local_pose_callback(self, msg):
        self.local_pose = msg
        self.local_q = self.quat_from_pose(msg)

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

    def takeoff_detection(self):
        cur = np.array([self.local_pose.pose.position.x,
                        self.local_pose.pose.position.y,
                        self.local_pose.pose.position.z])
        dist = np.linalg.norm(cur - self.x0[:3])
        return dist < 0.5 and self.arm_state

    def start(self):
        if not self.check_connection():
            print("Failed to connect!")
            return

        # takeoff to reach desired tracking height
        if not self.takeoff():
            print("Failed to takeoff!")
            return
        print("Successful Takeoff!")
        self.t0 = rospy.get_rostime().to_sec()

        rate = rospy.Rate(int(1 / self.dt))  # Hz

        while self.arm_state and not rospy.is_shutdown():
            # takeoff_pos = self.construct_pose_target(x=0, y=0, z=self.takeoff_height, q=self.takeoff_q)
            # count = 0
            # self.pos_control_pub.publish(takeoff_pos)

            cur_p = np.array([
                self.local_pose.pose.position.x,
                self.local_pose.pose.position.y,
                self.local_pose.pose.position.z
            ])
            if self.reached_target(cur_p, target_p=self.xg[:3], threshold=0.3):
                print("Reached Target!")
                break

            if self.mavros_state == State.MODE_PX4_OFFBOARD:
                time_since_solved = time.time() - self.t_solved
                # need_new_plan = time_since_solved > self.T
                need_new_plan = True
                if self.solver_thread is None or (not self.solver_thread.is_alive() and need_new_plan):
                    print("Generating action!")
                    # Spawn a process to run this independently:
                    self.solver_thread = threading.Thread(target=self.generate_action)
                    self.solver_thread.start()

                if not self.solved:
                    # print("Staying at origin!")
                    desired_pos = self.construct_pose_target(x=self.x0[0], y=self.x0[1], z=self.x0[2])
                    self.pos_control_pub.publish(desired_pos)
                else:
                    # print("Using prev path!")
                    # if self.version == 1:
                    #     u = self.U_mpc[path_index] + self.Gff
                    #     x = self.X_mpc[path_index]
                    #     thrust, phid, thetad, psid = self.drone_mpc.inverse_dyn(self.local_q, x_ref=x, u=u)
                    #     target_q = Rotation.from_euler("XYZ", [phid, thetad, psid]).as_quat()
                    #     self.control_attitude(target_q=target_q, thrust=thrust)
                    # else:
                    self.pos_control_pub.publish(self.use_prev_traj())
                    self.path_index += 1

                if self.planner.trajectory is not None:
                    self.output_path_pub.publish(
                        utils.create_path(traj=self.planner.trajectory, dt=self.dt, frame="world"))

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass


if __name__ == '__main__':
    con = Px4Controller()
    con.start()
