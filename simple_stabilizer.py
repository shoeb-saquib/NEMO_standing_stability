import mujoco as mj
import numpy as np
from mujoco_utils import MujocoUtils


def get_max_clip_in_floor(frame, corners, robot_center, threshold):
    error = None
    for corner in corners:
        transformed_corner = frame @ (corner - robot_center)
        if transformed_corner[2] < threshold:
            if not error:
                error = transformed_corner[2]
            else:
                if transformed_corner[2] < error:
                    error = transformed_corner[2]
    return error


class SimpleStabilizer:

    def __init__(self):
        self.model = mj.MjModel.from_xml_path("models/nemo/nemo5_nostl.xml")
        self.data = mj.MjData(self.model)
        self.kp = 3

        self.m = np.sum(self.model.body_mass)
        self.g = 9.81
        self.com = self.data.subtree_com[0]

        self.pitch_torque = 0
        self.roll_torque = 0
        self.hip_yaw_pos = 0

    def calculate_pitch_torque(self, pz_req):
        torque = 0.5 * self.m * self.g * pz_req
        return self.kp * torque

    def calculate_roll_torque(self, pz_req):
        torque = -0.5 * self.m * self.g * pz_req
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

    def make_floor_frame_from_foot(self, foot_prefix="left"):
        ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"{foot_prefix}_foot_{i}") for i in range(1, 5)]
        corners = np.stack([self.data.geom_xpos[i] for i in ids])
        foot_normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        z_hat = -foot_normal
        z_hat /= np.linalg.norm(z_hat)
        x_hat = np.array([z_hat[2], 0.0, -z_hat[0]])
        x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        frame = np.vstack([x_hat, y_hat, z_hat])
        return frame, corners

    def estimate_floor_frame_from_feet(self, robot_center):
        left_frame, left_corners = self.make_floor_frame_from_foot("left")
        right_frame, right_corners = self.make_floor_frame_from_foot("right")
        threshold = -0.08
        left_error = get_max_clip_in_floor(left_frame, right_corners, robot_center, threshold)
        if not left_error:
            return left_frame
        right_error = get_max_clip_in_floor(right_frame, left_corners, robot_center, threshold)
        if not right_error or left_error < right_error:
            return right_frame
        else:
            return left_frame

    def get_pitch_torque(self):
        return self.pitch_torque

    def get_roll_torque(self):
        return self.roll_torque

    def get_hip_yaw_pos(self):
        return self.hip_yaw_pos

    def step(self, positions, sensor_data, desired_com_offset):
        self.data.qpos[:7] = [0] * 7
        self.data.qpos[7:] = positions
        mj.mj_step(self.model, self.data)
        robot_center = self.calculate_robot_center()
        frame = self.estimate_floor_frame_from_feet(robot_center)
        com = frame @ (self.com - robot_center)
        com = np.array([com[0], com[1], com[2] + 0.14])
        desired_com = frame @ np.array([desired_com_offset, 0, 0])
        omega2 = self.g / com[2]
        desired_accel_x = (-50 * (com[0] - desired_com[0])) / self.m
        desired_accel_y = (-50 * (com[1] - desired_com[1])) / self.m
        pz_req = [com[0] - desired_accel_x / omega2, com[1] - desired_accel_y / omega2]
        self.pitch_torque = self.calculate_pitch_torque(pz_req[0])
        self.roll_torque = self.calculate_roll_torque(pz_req[1])
        self.hip_yaw_pos = self.calculate_hip_yaw_pos(sensor_data)