import mujoco as mj
import numpy as np
from mujoco_utils import MujocoUtils


class SimpleAnkleController:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.kp = 2
        self.kd = 1

        self.dt = model.opt.timestep
        self.m = np.sum(model.body_mass)
        self.g = abs(model.opt.gravity[2])
        self.com = data.subtree_com[0]
        self.prev_com = self.com.copy()
        self.robot_center = [0, 0, 0]
        self.pz_req = [0, 0]

        self.l_pitch_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "l_foot_pitch")
        self.r_pitch_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "r_foot_pitch")
        self.l_pitch_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "l_foot_pitch")
        self.r_pitch_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "r_foot_pitch")
        self.l_pitch_body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "l_foot_pitch")
        self.l_roll_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "l_foot_roll")
        self.r_roll_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "r_foot_roll")


        self.com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "com_marker")
        self.desired_com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "desired_com_marker")
        self.pz_req_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pz_req_marker")
        self.pz_curr_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pz_curr_marker")

    def apply_pitch_torque(self, pz_req):
        torque = 0.5 * self.m * self.g * (pz_req - self.data.xpos[self.l_pitch_body][0])
        self.data.ctrl[self.l_pitch_act] = self.kp * torque - self.kd * self.data.qvel[self.l_pitch_joint]
        self.data.ctrl[self.r_pitch_act] = self.kp * torque - self.kd * self.data.qvel[self.r_pitch_joint]

    def apply_roll_torque(self, pz_req):
        torque = -0.5 * self.m * self.g * (pz_req - self.robot_center[1])
        self.data.ctrl[self.l_roll_act] = self.kp * torque
        self.data.ctrl[self.r_roll_act] = self.kp * torque

    def show_debug_markers(self, desired_com):
        com = self.com
        self.data.site_xpos[self.com_marker] = np.array([com[0], com[1], com[2]])
        self.data.site_xpos[self.desired_com_marker] = np.array([desired_com[0], desired_com[1], com[2]])
        self.data.site_xpos[self.pz_req_marker] = np.array([self.pz_req[0], self.pz_req[1], 0])
        self.data.site_xpos[self.pz_curr_marker] = np.array([self.data.xpos[self.l_pitch_body][0], self.robot_center[1], 0])

    def step(self, desired_com, robot_center):
        self.robot_center = robot_center
        com_vel = (self.com - self.prev_com) / self.dt
        self.prev_com = self.com.copy()
        omega2 = self.g / self.com[2]
        desired_accel_x = (-50 * (self.com[0] - desired_com[0]) - 5 * com_vel[0]) / self.m
        desired_accel_y = (-50 * (self.com[1] - desired_com[1]) - 5 * com_vel[1]) / self.m
        self.pz_req = [self.com[0] - desired_accel_x / omega2, self.com[1] - desired_accel_y / omega2]
        self.apply_pitch_torque(self.pz_req[0])
        self.apply_roll_torque(self.pz_req[1])
