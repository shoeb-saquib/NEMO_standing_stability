import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

def get_max_floor_clip(floor_frame, foot_corners, contact_center, threshold):
    """

    :param floor_frame:
    :param foot_corners:
    :param contact_center:
    :param threshold:
    :return:
    """
    error = None
    for corner in foot_corners:
        transformed_corner = floor_frame @ (corner - contact_center)
        if transformed_corner[2] < threshold:
            if not error:
                error = transformed_corner[2]
            else:
                if transformed_corner[2] < error:
                    error = transformed_corner[2]
    return error

def estimate_base_velocity(foot_jac, joint_vel):
    """
    Calculate linear velocity of base with the assumption that the foot is stationary.

    :param foot_jac:
    :param joint_vel:
    :return:
    """

    jac_base = foot_jac[:, :6]
    jac_joints = foot_jac[:, 6:]
    foot_vel_from_joints = jac_joints @ joint_vel
    base_vel = -np.linalg.solve(jac_base, foot_vel_from_joints)
    return base_vel[0:3]

def exponential_filter(previous, current, alpha):
    return alpha * previous + (1 - alpha) * current

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

class Stabilizer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.m = np.sum(model.body_mass)
        self.g = np.array([0, 0, -9.81])
        self.linear_force_to_accel = np.hstack((np.eye(3) / self.m, np.zeros((3, 3))))
        self.com = data.subtree_com[0]
        self.prev_vel = np.zeros(3)
        self.left_foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_foot_roll")
        self.left_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "right_foot")

    def get_jacobian(self, site):
        nv = self.model.nv
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        mj.mj_jacSite(self.model, self.data, jp, jr, site)
        return np.vstack((jp, jr))

    def get_contact_center(self):
        l_foot_center = self.data.site_xpos[self.left_site]
        r_foot_center = self.data.site_xpos[self.right_site]
        return np.array([l_foot_center[0], (r_foot_center[1] + l_foot_center[1]) / 2, l_foot_center[2]])

    def get_foot_corners(self, foot_prefix="left"):
        ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"{foot_prefix}_foot_{i}") for i in range(1, 5)]
        return np.stack([self.data.geom_xpos[i] for i in ids])

    def make_floor_frame_from_foot(self, foot_prefix="left"):
        corners = self.get_foot_corners(foot_prefix=foot_prefix)
        foot_normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        z_hat = -foot_normal
        z_hat /= np.linalg.norm(z_hat)
        x_hat = corners[0] - corners[2]
        x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        frame = np.vstack([x_hat, y_hat, z_hat])
        return frame, corners

    def estimate_floor_frame_from_feet(self, contact_center):
        left_frame, left_corners = self.make_floor_frame_from_foot("left")
        right_frame, right_corners = self.make_floor_frame_from_foot("right")
        threshold = -0.08
        left_error = get_max_floor_clip(left_frame, right_corners, contact_center, threshold)
        if not left_error:
            return left_frame
        right_error = get_max_floor_clip(right_frame, left_corners, contact_center, threshold)
        if not right_error or left_error < right_error:
            return right_frame
        else:
            return left_frame

    def update_simulation(self, joint_pos, joint_vel):
        # Set world frame to base orientation and base velocity to zero
        self.data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
        self.data.qvel[:6] = np.zeros(6)

        # Update joint state
        self.data.qpos[7:] = joint_pos
        self.data.qvel[6:] = joint_vel
        mj.mj_forward(self.model, self.data)

        # Rebase world frame to left foot
        foot_position = self.data.body("l_foot_roll").xpos
        foot_to_world = self.data.body("l_foot_roll").xmat.reshape(3,3)
        world_to_foot = foot_to_world.T
        base_position = -world_to_foot @ foot_position
        base_quat = quat_from_mat(foot_to_world)
        self.data.qpos[0:3] = base_position
        self.data.qpos[3:7] = base_quat
        mj.mj_forward(self.model, self.data)

    def compute_inertia_matrix(self, floor_frame):
        sum_m_d2 = 0.0
        for i in range(self.model.nbody):
            mi = self.model.body_mass[i]
            ri = self.data.xipos[i]
            di2 = np.dot(ri - self.com, ri - self.com)
            sum_m_d2 += mi * di2

        inertia_axis = (2.0 / 3.0) * sum_m_d2
        inertia_world = np.eye(3) * inertia_axis
        inertia_floor = floor_frame @ inertia_world @ floor_frame.T
        return inertia_floor

    def construct_force_to_accel_matrix(self, floor_frame, inertia_matrix, left=True):
        if left: foot_site = self.left_site
        else: foot_site = self.right_site
        com_relative_to_foot = floor_frame @ (self.com - self.data.site_xpos[foot_site])
        skew_x = skew(com_relative_to_foot)
        inertia_inv = np.linalg.inv(inertia_matrix)
        a = np.vstack((self.linear_force_to_accel, np.hstack((inertia_inv @ skew_x, inertia_inv))))
        if left: a = np.hstack((a, np.zeros((6, 6))))
        else: a = np.hstack((np.zeros((6, 6)), a))
        return a

    def compute_desired_contact_forces(self, floor_frame, desired_linear_accel, desired_angular_accel):
        inertia_matrix = self.compute_inertia_matrix(floor_frame)
        left_force_to_accel = self.construct_force_to_accel_matrix(floor_frame, inertia_matrix, left=True)
        right_force_to_accel = self.construct_force_to_accel_matrix(floor_frame, inertia_matrix, left=False)
        force_to_accel = left_force_to_accel + right_force_to_accel
        desired_accel = np.hstack((desired_linear_accel - self.g, desired_angular_accel)).reshape((6, 1))
        return np.linalg.pinv(force_to_accel) @ desired_accel

    def calculate_joint_torques(self, joint_pos, joint_vel, relative_desired_com, sensor_data):
        self.update_simulation(joint_pos, joint_vel)

        jac_left = self.get_jacobian(self.left_site)
        jac_right = self.get_jacobian(self.right_site)

        base_vel = estimate_base_velocity(jac_left, joint_vel)
        com_vel = exponential_filter(self.prev_vel, base_vel, 0)

        sensor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer")
        start = self.model.sensor_adr[sensor_id]
        accel = np.array(sensor_data[start:start + 3])

        self.prev_vel = com_vel

        contact_center = self.get_contact_center()
        floor_frame = self.estimate_floor_frame_from_feet(contact_center)
        relative_com = floor_frame @ (self.com - contact_center)
        desired_linear_accel = 10 * (relative_desired_com - relative_com) - 0.3 * com_vel

        q = self.data.xquat[self.left_foot_id]

        # scipy needs quaternion in the form [x, y, z, w]
        rotvec = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()

        desired_angular_accel = 100 * rotvec

        contact_forces = self.compute_desired_contact_forces(floor_frame, desired_linear_accel, desired_angular_accel)
        left_force = np.concatenate((floor_frame.T @ contact_forces[:3, 0], floor_frame.T @ contact_forces[3:6, 0]))
        right_force = np.concatenate((floor_frame.T @ contact_forces[6:9, 0], floor_frame.T @ contact_forces[9:, 0]))
        joint_torques = -(jac_left.T @ left_force + jac_right.T @ right_force)
        return joint_torques[6:], base_vel


