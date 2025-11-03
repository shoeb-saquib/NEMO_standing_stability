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

def skew(v):
    x, y, z = v
    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

class OmniscientStabilizer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.m = np.sum(model.body_mass)
        self.g = np.array([0, 0, -9.81, 0, 0, 0]).reshape((6, 1))
        self.a_top = np.hstack((np.eye(3) / self.m, np.zeros((3, 3))))
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

    def compute_inertia_matrix(self):
        sum_m_d2 = 0.0
        for i in range(self.model.nbody):
            mi = self.model.body_mass[i]
            ri = self.data.xipos[i]  # 3-vector
            di2 = np.dot(ri - self.com, ri - self.com)
            sum_m_d2 += mi * di2

        if self.m <= 0:
            return np.eye(3) * 1e-6

        i_axis = (2.0 / 3.0) * sum_m_d2
        i_com = np.eye(3) * i_axis
        return i_com

    def construct_coefficient_matrix(self, inertia_matrix, left=True):
        if left: foot_site = self.left_site
        else: foot_site = self.right_site
        x = self.com - self.data.site_xpos[foot_site]
        skew_x = skew(x)
        inv_i = np.linalg.inv(inertia_matrix)
        a = np.vstack((self.a_top, np.hstack((inv_i @ skew_x, inv_i))))
        if left: a = np.hstack((a, np.zeros((6, 6))))
        else: a = np.hstack((np.zeros((6, 6)), a))
        return a

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
        q = quat_conjugate(self.data.xquat[self.pelvis_id])

        # scipy needs quaternion in the form [x, y, z, w]
        rotvec = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
        desired_angular_accel = 100 * rotvec - 10 * self.data.qvel[3:6]

        i = self.compute_inertia_matrix()
        a = self.construct_coefficient_matrix(i, True) + self.construct_coefficient_matrix(i, False)
        b = np.hstack((desired_accel, desired_angular_accel)).reshape((6, 1)) - self.g
        f = np.linalg.pinv(a) @ b
        torques = -(jl.T @ f[:6, 0] + jr.T @ f[6:, 0])
        values = [np.linalg.norm(self.com), np.linalg.norm(desired_com - self.com), np.linalg.norm(com_vel), np.linalg.norm(desired_accel),
                  np.linalg.norm(rotvec), np.linalg.norm(desired_angular_accel)]
        return torques[6:], values


