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

def skew(v):
    x, y, z = v
    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

def R_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def R_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

class Stabilizer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.m = np.sum(model.body_mass)
        self.g = np.array([0, 0, -9.81, 0, 0, 0]).reshape((6, 1))
        self.a_top = np.hstack((np.eye(3) / self.m, np.zeros((3, 3))))
        self.com = data.subtree_com[0]
        self.prev_com = None
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

    def compute_inertia_matrix(self, frame):
        sum_m_d2 = 0.0
        for i in range(self.model.nbody):
            mi = self.model.body_mass[i]
            ri = self.data.xipos[i]
            di2 = np.dot(ri - self.com, ri - self.com)
            sum_m_d2 += mi * di2

        if self.m <= 0:
            i_world = np.eye(3) * 1e-6
        else:
            i_axis = (2.0 / 3.0) * sum_m_d2
            i_world = np.eye(3) * i_axis

        i_floor = frame @ i_world @ frame.T
        return i_floor

    def construct_coefficient_matrix(self, frame, inertia_matrix, left=True):
        if left: foot_site = self.left_site
        else: foot_site = self.right_site
        x = frame @ (self.com - self.data.site_xpos[foot_site])
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

    def calculate_joint_torques(self, dt, positions, desired_offset, true_com_vel):
        self.data.qpos[:7] = [0] * 7
        self.data.qpos[7:] = positions
        mj.mj_step(self.model, self.data)
        jl = self.get_jacobian(self.left_site)
        jr = self.get_jacobian(self.right_site)
        robot_center = self.calculate_robot_center()
        frame,_ = self.make_floor_frame_from_foot()
        com = frame @ (self.com - robot_center)
        # if self.prev_com is not None:
        #     com_vel = (com - self.prev_com)/ dt
        # else:
        #     com_vel = np.zeros(3)
        # self.prev_com = com
        # alpha = 0.9
        # com_vel = alpha * self.prev_com_vel + (1 - alpha) * com_vel
        # self.prev_com_vel = com_vel
        # qvel_copy = qvel.copy()
        # qvel_copy[:6] = 0
        # com_vel = -(jl @ qvel_copy)[:3]
        curr = R_y(positions[4]) @ R_x(positions[5]) @ np.array([0.0, 0.0, com[2]])
        com_vel = np.zeros(3)
        if self.prev_com is not None: com_vel = (curr - self.prev_com) / dt
        self.prev_com = curr
        desired_accel = 10 * (desired_offset - com) - 0*4 * com_vel
        #desired_accel = np.clip(desired_accel, -1, 1)
        q = self.data.xquat[self.left_foot_id]

        # scipy needs quaternion in the form [x, y, z, w]
        rotvec = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
        omega = self.data.cvel[self.left_foot_id][:3]
        desired_angular_accel = 100 * rotvec - 0*10 * omega
        i = self.compute_inertia_matrix(frame)
        a = self.construct_coefficient_matrix(frame, i, True) + self.construct_coefficient_matrix(frame, i, False)
        b = np.hstack((desired_accel, desired_angular_accel)).reshape((6, 1)) - self.g
        f = np.linalg.pinv(a) @ b

        fl = np.concatenate((frame.T @ f[:3, 0], frame.T @ f[3:6, 0]))
        fr = np.concatenate((frame.T @ f[6:9, 0], frame.T @ f[9:, 0]))
        torques = -(jl.T @ fl+ jr.T @ fr)
        values = com_vel
        return torques[6:], values


