import mujoco as mj
import numpy as np


class SimpleAnkleController:

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.dt = model.opt.timestep
        self.m = np.sum(model.body_mass)
        self.g = abs(model.opt.gravity[2])
        self.com = data.subtree_com[0]
        self.prev_com = self.com.copy()

        self.l_pitch = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "l_foot_pitch")
        self.r_pitch = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "r_foot_pitch")

        self.com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "com_marker")
        self.desired_com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "desired_com_marker")
        self.pz_req_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pz_req_marker")
        self.pz_curr_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pz_curr_marker")

    def body_pos_xy(self, body_name):
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        pos = self.data.xpos[bid]
        return np.array([pos[0], pos[1]])

    def calculate_ankle_torque(self, pz_req, body_name):
        return 0.5 * self.m * self.g * (pz_req - self.body_pos_xy(body_name)[0])

    def apply_ankle_torque(self, pz_req):
        torque = self.calculate_ankle_torque(pz_req, "l_foot_pitch")
        self.data.ctrl[self.l_pitch] = torque
        self.data.ctrl[self.r_pitch] = torque

    def show_debug_markers(self, desired_com):
        com = self.com
        self.data.site_xpos[self.com_marker] = np.array([com[0], com[1], com[2]])
        self.data.site_xpos[self.desired_com_marker] = np.array([desired_com, com[1], com[2]])
        self.data.site_xpos[self.pz_req_marker] = np.array([self.pz_req, com[1], 0])
        self.data.site_xpos[self.pz_curr_marker] = np.array([self.body_pos_xy("l_foot_pitch")[0], com[1], 0])

    def step(self, desired_com):
        com_vel = (self.com - self.prev_com) / self.dt
        omega2 = self.g / self.com[2]
        desired_com_acceleration = (-50 * (self.com[0] - desired_com) - 5 * com_vel[0]) / self.m
        self.pz_req = self.com[0] - desired_com_acceleration / omega2
        self.apply_ankle_torque(self.pz_req)
        self.prev_com = self.com.copy()
