import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

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

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def rotation_vector_from_quat_error(q_des, q_cur, eps=1e-8):
    qerr = quat_mul(q_des, quat_conjugate(q_cur))
    qerr = qerr / np.linalg.norm(qerr)

    w = np.clip(qerr[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(0.0, 1.0 - w * w))

    if s < eps or angle < eps:
        return 2.0 * qerr[1:]
    else:
        axis = qerr[1:] / s
        return axis * angle

class OmniscientStabilizer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.m = np.sum(model.body_mass)
        self.fg = self.m * np.array([0, 0, -9.81])
        self.com = data.subtree_com[0]
        self.prev_com = self.com.copy()
        self.pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
        self.left_foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_foot_roll")
        self.left_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "right_foot")

    def get_jacobian(self, site):
        nv = self.model.nv
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        mj.mj_jacSite(self.model, self.data, jp, jr, site)
        return np.vstack((jp, jr))

    def compute_I_com(self):
        sum_m_d2 = 0.0
        for i in range(self.model.nbody):
            mi = self.model.body_mass[i]
            ri = self.data.xipos[i]  # 3-vector
            di2 = np.dot(ri - self.com, ri - self.com)
            sum_m_d2 += mi * di2

        if self.m <= 0:
            return np.eye(3) * 1e-6

        I_axis = (2.0 / 3.0) * sum_m_d2
        I_com = np.eye(3) * I_axis
        return I_com

    def calculate_robot_center(self):
        l_foot_center = self.data.site_xpos[self.left_site]
        r_foot_center = self.data.site_xpos[self.right_site]
        return np.array([l_foot_center[0], (r_foot_center[1] + l_foot_center[1]) / 2, l_foot_center[2]])

    def calculate_joint_torques(self, dt, desired_offset):
        robot_center = self.calculate_robot_center()
        desired_com = desired_offset + robot_center
        jl = self.get_jacobian(self.left_site)
        jr = self.get_jacobian(self.right_site)
        com_vel = (self.com - self.prev_com) / dt
        self.prev_com = self.com.copy()
        desired_accel = 10 * (desired_com - self.com) - 4 * com_vel
        desired_accel = np.clip(desired_accel, -5, 5)
        q = quat_conjugate(self.data.xquat[self.pelvis_id])

        # scipy needs quaternion in the form [x, y, z, w]
        rotvec = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
        desired_angular_accel = 100 * rotvec - 10 * self.data.qvel[3:6]
        f = (self.m * desired_accel - self.fg) / 2
        t = (self.compute_I_com() @ desired_angular_accel) / 2
        wrench = np.concatenate((f, t))
        torques = -(jl.T @ wrench + jr.T @ wrench)
        values = [np.linalg.norm(self.com), np.linalg.norm(desired_com - self.com), np.linalg.norm(com_vel), np.linalg.norm(desired_accel),
                  np.linalg.norm(rotvec), np.linalg.norm(desired_angular_accel)]
        return torques[6:], values


