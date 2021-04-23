import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path


def create_path(traj, dt, frame="world"):
    poses = []
    cur_tstamp = rospy.Time.now()
    for i in range(traj.shape[0]):
        tstamp = cur_tstamp + rospy.Duration(dt * i)
        p = PoseStamped()

        p.pose.position = Point(*traj[i, :3])

        p.header.frame_id = frame
        p.header.stamp = tstamp
        poses.append(p)

    path = Path()
    path.header.frame_id = frame
    path.header.stamp = cur_tstamp
    path.poses = poses
    return path
