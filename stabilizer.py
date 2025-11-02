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

def quat_from_mat(r):
    qw = np.sqrt(1.0 + np.trace(r)) / 2.0
    qx = (r[2, 1] - r[1, 2]) / (4.0 * qw)
    qy = (r[0, 2] - r[2, 0]) / (4.0 * qw)
    qz = (r[1, 0] - r[0, 1]) / (4.0 * qw)
    return np.array([qw, qx, qy, qz])

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

class Stabilizer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.m = np.sum(model.body_mass)
        self.fg = self.m * np.array([0, 0, -9.81])
        self.com = data.subtree_com[0]
        self.prev_com = self.com.copy()
        self.prev_frame = None
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

    def compute_I_com(self, frame):
        sum_m_d2 = 0.0
        for i in range(self.model.nbody):
            mi = self.model.body_mass[i]
            ri = self.data.xipos[i]
            di2 = np.dot(ri - self.com, ri - self.com)
            sum_m_d2 += mi * di2

        if self.m <= 0:
            I_world = np.eye(3) * 1e-6
        else:
            I_axis = (2.0 / 3.0) * sum_m_d2
            I_world = np.eye(3) * I_axis

        I_floor = frame @ I_world @ frame.T
        return I_floor

    def calculate_robot_center(self):
        l_foot_center = self.data.site_xpos[self.left_site]
        r_foot_center = self.data.site_xpos[self.right_site]
        return np.array([l_foot_center[0], (r_foot_center[1] + l_foot_center[1]) / 2, l_foot_center[2]])

    def make_floor_frame_from_foot(self, foot_prefix="left"):
        ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"{foot_prefix}_foot_{i}") for i in range(1, 5)]
        corners = np.stack([self.data.geom_xpos[i] for i in ids])
        foot_normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        z_hat = -foot_normal
        z_hat /= np.linalg.norm(z_hat)
        x_hat = corners[0] - corners[2]
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

    def calculate_joint_torques(self, dt, positions, desired_offset, omega):
        self.data.qpos[:7] = [0] * 7
        self.data.qpos[7:] = positions
        mj.mj_step(self.model, self.data)
        robot_center = self.calculate_robot_center()
        frame = self.estimate_floor_frame_from_feet(robot_center)
        if self.prev_frame is None:
            self.prev_frame = frame
        com = frame @ (self.com - robot_center)
        desired_com = frame @ desired_offset
        com_vel = (frame @ self.com - self.prev_frame @ self.prev_com) / dt
        self.prev_com = self.com.copy()
        desired_accel = 10 * (desired_com - com) - 4 * com_vel
        desired_accel = np.clip(desired_accel, -5, 5)
        q = self.data.xquat[self.left_foot_id]

        #scipy needs quaternion in the form [x, y, z, w]
        rotvec = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
        #omega = self.data.cvel[self.left_foot_id][:3]
        desired_angular_accel = 100 * rotvec - 10 * omega
        f = (self.m * desired_accel - self.fg) / 2
        f = frame.T @ f
        t = (self.compute_I_com(frame) @ desired_angular_accel) / 2
        t = frame.T @ t
        wrench = np.concatenate((f, t))
        jl = self.get_jacobian(self.left_site)
        jr = self.get_jacobian(self.right_site)
        torques = -(jl.T @ wrench + jr.T @ wrench)
        self.prev_frame = frame
        values = [np.linalg.norm(com), np.linalg.norm(desired_com - com), np.linalg.norm(com_vel), np.linalg.norm(desired_accel),
                  np.linalg.norm(rotvec), np.linalg.norm(desired_angular_accel)]
        return torques[6:], values


