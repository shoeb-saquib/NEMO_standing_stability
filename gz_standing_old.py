import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
import time
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Duration, Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from gz_sim_interfaces.msg import StateObservationReduced
from gz_sim_interfaces.msg import KeyboardCmd
from geometry_msgs.msg import Twist

helper_path = os.path.join(
    get_package_share_directory('prairie_control'),
    "helpers")

sys.path.append(helper_path)
import utils

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint",
                       "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint",
                       "r_foot_roll_joint",
                       "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_joint",
                       "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_joint"]


class gz_standing(Node):
    def __init__(self):
        super().__init__('gz_standing')
        qos_profile = QoSProfile(depth=10)

        # Read IMU data and publish JTP

        self.home_pose = np.array([
            -0.698132, 0, 0, 1.22173, -0.523599, 0,
            -0.698132, 0, 0, 1.22173, -0.523599, 0,
            0, 0.05, 0, 0, -0.05, 0
        ])

        self.ff_torque = np.array([
            -0.6, 0.0, 0.0, 1.11, -0.3, 0.0,
            -0.6, 0.0, 0.0, 1.11, -0.3, 0.0,
            0., 0., 0., 0., 0., 0.
        ])

        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/gz_state_observation',
            self.state_callback,
            qos_profile
        )

        self.ankle_l_id = 4
        self.ankle_r_id = 10

        self.joint_pub = self.create_publisher(JointTrajectory, 'gz_standing_jtp', qos_profile)
        self.timer = self.create_timer(0.002, self.timer_callback)
        self.obs = {}

    def state_callback(self, msg):
        self.obs = utils.fill_obs_dict(msg)
        return

    def timer_callback(self):
        joint_traj = JointTrajectory()
        jtp = JointTrajectoryPoint()
        jtp2 = JointTrajectoryPoint()

        now = self.get_clock().now()
        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST_COMPLETE

        jtp.time_from_start = Duration()
        jtp.time_from_start.sec = 0
        jtp.time_from_start.nanosec = 0

        jtp2.time_from_start = Duration()
        jtp2.time_from_start.sec = 0
        jtp2.time_from_start.nanosec = 0

        tau_delta = self.ff_torque.copy()

        pos_t = self.home_pose.copy()

        if self.obs != {}:
            time_coeff = min(self.obs["time"] / 0.5, 1.0)
            des_pos = time_coeff * self.home_pose
            jtp_ankle_ff = self.obs["linear_acceleration"][0] * -2.5

            left_ankle_pos = self.obs["joint_position"][self.ankle_l_id]
            right_ankle_pos = self.obs["joint_position"][self.ankle_r_id]

            # left_pd = 10 * (des_pos - left_ankle_pos) - 5 * self.obs["joint_velocity"][self.ankle_l_id]
            # right_pd = 10 * (des_pos - right_ankle_pos) - 5 * self.obs["joint_velocity"][self.ankle_r_id]

            tau_delta[self.ankle_l_id] += jtp_ankle_ff
            tau_delta[self.ankle_r_id] += jtp_ankle_ff
            pos_t *= time_coeff

            # Plus additional PD

            gain = 0
            damping = 0

            pd2 = (des_pos - self.obs["joint_position"]) * gain - self.obs["joint_velocity"] * damping
            tau_delta += pd2

        jtp.positions = pos_t.tolist()
        jtp.velocities = [0.] * 18
        jtp.effort = tau_delta.tolist()

        jtp2.positions = pos_t.tolist()
        jtp2.velocities = [0.] * 18
        jtp2.effort = tau_delta.tolist()
        jtp2.time_from_start.sec = 100
        joint_traj.points = [jtp, jtp2]
        # print(tau_delta)
        self.joint_pub.publish(joint_traj)


def main(args=None):
    rclpy.init(args=args)

    node = gz_standing()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()