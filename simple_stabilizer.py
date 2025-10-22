import mujoco as mj
import numpy as np
from mujoco_utils import MujocoUtils


class SimpleStabilizer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.kp = 3

        self.m = np.sum(model.body_mass)
        self.g = 9.81
        self.com = data.subtree_com[0]

        self.pitch_torque = 0
        self.roll_torque = 0
        self.hip_yaw_pos = 0

    def calculate_pitch_torque(self, pz_req, robot_center):
        torque = 0.5 * self.m * self.g * (pz_req - robot_center[0])
        return self.kp * torque

    def calculate_roll_torque(self, pz_req, robot_center):
        torque = -0.5 * self.m * self.g * (pz_req - robot_center[1])
        return self.kp * torque

    def calculate_robot_center(self):
        l_foot_center = MujocoUtils.body_pos(self.model, self.data, "l_foot_pitch")
        r_foot_center = MujocoUtils.body_pos(self.model, self.data, "r_foot_pitch")
        return [l_foot_center[0], (r_foot_center[1] + l_foot_center[1]) / 2, l_foot_center[2]]

    def calculate_hip_yaw_pos(self, sensor_data):
        sensor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer")
        start = self.model.sensor_adr[sensor_id]
        accel = np.array(sensor_data[start:start + 3])
        accel /= np.linalg.norm(accel)
        yaw = np.arctan2(accel[1], accel[2])
        yaw = np.clip(yaw, -1, 1)
        return yaw

    def make_floor_frame_from_foot(self):
        ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"left_foot_{i}") for i in range(1, 5)]
        corners = np.stack([self.data.geom_xpos[i] for i in ids])
        foot_normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        foot_normal /= np.linalg.norm(foot_normal)
        z_hat = -foot_normal
        torso_forward = np.array([1.0, 0.0, 0.0])
        x_hat = torso_forward - np.dot(torso_forward, z_hat) * z_hat
        x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        r_floor = np.column_stack([x_hat, y_hat, z_hat])
        return r_floor

    def step(self, desired_com_offset, sensor_data):
        r_floor = self.make_floor_frame_from_foot()
        com = r_floor.T @ (self.com - self.calculate_robot_center())
        com = np.array([com[0], com[1], com[2] + 0.14])
        desired_com = r_floor.T @ np.array([desired_com_offset, 0, 0])
        omega2 = self.g / com[2]
        desired_accel_x = (-50 * (com[0] - desired_com[0])) / self.m
        desired_accel_y = (-50 * (com[1] - desired_com[1])) / self.m
        pz_req = [com[0] - desired_accel_x / omega2, com[1] - desired_accel_y / omega2]
        self.pitch_torque = self.calculate_pitch_torque(pz_req[0], [0, 0, 0])
        self.roll_torque = self.calculate_roll_torque(pz_req[1], [0, 0, 0])
        self.hip_yaw_pos = self.calculate_hip_yaw_pos(sensor_data)