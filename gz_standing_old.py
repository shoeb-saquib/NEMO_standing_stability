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

helper_path = os.path.join(get_package_share_directory('prairie_control'), "helpers")
sys.path.append(helper_path)
import utils
import stabilizer

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint",
                       "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint",
                       "r_foot_roll_joint",
                       "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_joint",
                       "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_joint"]
SIT_TIME = 25
TRANSITION_TIME = 2


class gz_standing(Node):
    def __init__(self):
        super().__init__('gz_standing')
        qos_profile = QoSProfile(depth=10)

        # Read IMU data and publish JTP

        self.sit_pos = np.array([
            -0.9673, 0, 0, 2.13, -1.0735, 0,
            -0.9673, 0, 0, 2.13, -1.0735, 0,
            0, 0.05, 0, 0, -0.05, 0
        ])
        self.stand_pos = np.array([
            -1.06169705, 0, 0, 1.22173, -0.08, 0,
            -1.06169705, 0, 0, 1.22173, -0.08, 0,
            0, 0.05, 0, 0, -0.05, 0
        ])

        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/gz_state_observation',
            self.state_callback,
            qos_profile
        )

        self.num_joints = len(JOINT_LIST_COMPLETE)
        self.ids = {name: index for index, name in enumerate(JOINT_LIST_COMPLETE)}
        self.stabilizer = stabilizer.Stabilizer()

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

        tau_delta = np.zeros(self.num_joints)
        pos_t = self.sit_pos.copy()
        start_offset = 0.55
        end_offset = 0.22

        if self.obs != {}:
            t = self.obs["time"]
            if t <= SIT_TIME:
                com_offset = start_offset
                pos_t = self.sit_pos.copy()
            elif t <= SIT_TIME + TRANSITION_TIME:
                com_offset = stabilizer.lin_interp(t - SIT_TIME, TRANSITION_TIME, start_offset, end_offset)
                pos_t = stabilizer.lin_interp(t - SIT_TIME, TRANSITION_TIME, self.sit_pos, self.stand_pos)
            else:
                com_offset = end_offset
                pos_t = self.stand_pos.copy()
            pos_t[self.ids["l_foot_pitch_joint"]] = self.obs["joint_position"][self.ids["l_foot_pitch_joint"]]
            pos_t[self.ids["r_foot_pitch_joint"]] = self.obs["joint_position"][self.ids["r_foot_pitch_joint"]]
            pos_t[self.ids["l_foot_roll_joint"]] = self.obs["joint_position"][self.ids["l_foot_roll_joint"]]
            pos_t[self.ids["r_foot_roll_joint"]] = self.obs["joint_position"][self.ids["r_foot_roll_joint"]]
            self.stabilizer.step(self.obs["joint_position"], self.obs["linear_acceleration"], com_offset)
            tau_delta[self.ids["l_foot_pitch_joint"]] = self.stabilizer.get_pitch_torque()
            tau_delta[self.ids["r_foot_pitch_joint"]] = self.stabilizer.get_pitch_torque()
            tau_delta[self.ids["l_foot_roll_joint"]] = self.stabilizer.get_roll_torque()
            tau_delta[self.ids["r_foot_roll_joint"]] = self.stabilizer.get_roll_torque()
            tau_delta[self.ids["l_knee_joint"]] = -4 * self.obs["joint_velocity"][self.ids["l_knee_joint"]]
            tau_delta[self.ids["r_knee_joint"]] = -4 * self.obs["joint_velocity"][self.ids["r_knee_joint"]]
            if t > SIT_TIME + TRANSITION_TIME:
                pos_t[self.ids["l_hip_yaw_joint"]] = self.stabilizer.get_hip_yaw_pos()
                pos_t[self.ids["r_hip_yaw_joint"]] = self.stabilizer.get_hip_yaw_pos()

        jtp.positions = pos_t.tolist()
        jtp.velocities = [0.] * self.num_joints
        jtp.effort = tau_delta.tolist()

        jtp2.positions = pos_t.tolist()
        jtp2.velocities = [0.] * self.num_joints
        jtp2.effort = tau_delta.tolist()
        jtp2.time_from_start.sec = 100
        joint_traj.points = [jtp, jtp2]

        self.joint_pub.publish(joint_traj)


def main(args=None):
    rclpy.init(args=args)

    node = gz_standing()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()