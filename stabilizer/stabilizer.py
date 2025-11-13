import mujoco as mj
from scipy.spatial.transform import Rotation
from stabilizer.math_utils import *

GRAVITY = np.array([0, 0, -9.81])
TARGET_Z_ADJUSTMENT = -0.25


class Stabilizer:
    """
    Calculates joint torques to keep robot standing as long as the left foot is on the floor.
    """

    def __init__(self):
        self.model = mj.MjModel.from_xml_path("models/nemo/nemo5_nostl.xml")
        self.data = mj.MjData(self.model)

        self.m = np.sum(self.model.body_mass)
        self.linear_force_to_accel = np.hstack((np.eye(3) / self.m, np.zeros((3, 3))))

        self.com = self.data.subtree_com[0]
        self.prev_vel = np.zeros(3)

        self.left_foot_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "l_foot_roll")
        self.left_site = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_site = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "right_foot")

    def get_jacobian(self, site):
        """
        Compute the 6xN Jacobian matrix for a given site (3 linear, 3 angular).

        :param site: Integer site ID.
        :return: (6, nv) Jacobian matrix for that site.
        """
        nv = self.model.nv
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        mj.mj_jacSite(self.model, self.data, jp, jr, site)
        return np.vstack((jp, jr))

    def estimate_base_velocity(self, foot_jac):
        """
        Estimate the linear velocity of the robot base, assuming the foot is stationary.

        :param foot_jac: (6, N) Jacobian of the foot in world frame.
        :return: (3,) Estimated base linear velocity.
        """
        jac_base = foot_jac[:, :6]
        jac_joints = foot_jac[:, 6:]

        # Velocity of foot induced by joint motion
        foot_vel_from_joints = jac_joints @ self.data.qvel[6:]

        # Solve for base velocity assuming foot velocity = 0
        base_vel = -np.linalg.solve(jac_base, foot_vel_from_joints)
        return base_vel[0:3]

    def get_contact_center(self):
        """
        Compute the midpoint between foot sites (approximate contact center).

        :return: (3,) Contact center in world coordinates.
        """
        l_foot_center = self.data.site_xpos[self.left_site]
        r_foot_center = self.data.site_xpos[self.right_site]
        return np.array([
            l_foot_center[0],
            (r_foot_center[1] + l_foot_center[1]) / 2,
            l_foot_center[2]
        ])

    def get_foot_corners(self, foot_prefix="left"):
        """
        Get the positions of the foot corner geoms for constructing the local frame.

        :param foot_prefix: "left" or "right"
        :return: (4, 3) Array of corner positions.
        """
        ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"{foot_prefix}_foot_{i}") for i in range(1, 5)]
        return np.stack([self.data.geom_xpos[i] for i in ids])

    def construct_frame_from_foot(self, foot="left"):
        """
        Construct rotation matrix that maps world coordinates to foot frame.

        :param foot: "left" or "right"
        :return: (3, 3) World to floor rotation matrix.
        """
        corners = self.get_foot_corners(foot)

        # z-axis aligned with the foot's surface normal
        foot_normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        z_hat = -foot_normal

        # x-axis aligned along the side edge of the foot
        z_hat /= np.linalg.norm(z_hat)
        x_hat = corners[0] - corners[2]
        x_hat /= np.linalg.norm(x_hat)

        # y-axis perpendicular to x- and z-axis
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        return np.vstack([x_hat, y_hat, z_hat])

    def estimate_inertia_matrix(self, floor_frame):
        """
        Approximate the robot's inertia about the COM using a spherical model.

        :param floor_frame: (3, 3) Rotation matrix of floor frame.
        :return: (3, 3) Inertia matrix in floor coordinates.
        """
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
        """
        Build matrix that maps foot contact forces to COM linear and angular accelerations.

        :param floor_frame: (3, 3) World to floor rotation matrix.
        :param inertia_matrix: (3, 3) Inertia matrix in floor coordinates.
        :param left: Boolean, True for left foot, False for right.
        :return: (6, 12) Force-to-acceleration mapping matrix.
        """
        if left:
            foot_site = self.left_site
        else:
            foot_site = self.right_site
        x = floor_frame @ (self.com - self.data.site_xpos[foot_site])
        inertia_inv = np.linalg.inv(inertia_matrix)
        a = np.vstack((self.linear_force_to_accel, np.hstack((inertia_inv @ skew(x), inertia_inv))))
        if left:
            a = np.hstack((a, np.zeros((6, 6))))
        else:
            a = np.hstack((np.zeros((6, 6)), a))
        return a

    def compute_desired_contact_forces(self, floor_frame, desired_linear_accel, desired_angular_accel):
        """
        Compute foot contact forces necessary for desired accelerations.

        :param floor_frame: (3, 3) World to floor rotation matrix.
        :param desired_linear_accel: (3,) Desired COM linear acceleration.
        :param desired_angular_accel: (3,) Desired angular acceleration.
        :return: (12, 1) Desired contact forces for both feet.
        """
        inertia_matrix = self.estimate_inertia_matrix(floor_frame)
        left_force_to_accel = self.construct_force_to_accel_matrix(floor_frame, inertia_matrix, left=True)
        right_force_to_accel = self.construct_force_to_accel_matrix(floor_frame, inertia_matrix, left=False)
        force_to_accel = left_force_to_accel + right_force_to_accel
        desired_accel = np.hstack((desired_linear_accel - GRAVITY, desired_angular_accel)).reshape((6, 1))
        return np.linalg.pinv(force_to_accel) @ desired_accel

    def update_simulation(self, joint_pos, joint_vel):
        """
        Update MuJoCo simulation with given joint states and rebase to left foot frame.

        :param joint_pos: (nq-7,) Array of joint positions.
        :param joint_vel: (nv-6,) Array of joint velocities.
        """
        # Set world frame to base orientation
        self.data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
        # Set base velocity to 0
        self.data.qvel[:6] = np.zeros(6)

        # Update joint state
        self.data.qpos[7:] = joint_pos
        self.data.qvel[6:] = joint_vel
        mj.mj_forward(self.model, self.data)

        # Rebase world frame to left foot
        foot_to_world = self.data.body("l_foot_roll").xmat.reshape(3, 3)
        base_quat = quat_from_mat(foot_to_world)
        self.data.qpos[3:7] = base_quat
        mj.mj_forward(self.model, self.data)

    def get_relative_com(self):
        """
        Compute robot's current COM relative to contact center.

        :return: (3,) Relative COM.
        """
        return self.construct_frame_from_foot() @ (self.com - self.get_contact_center())

    def calculate_joint_torques(self, desired_com):
        """
        Compute joint torques to keep the relative COM at the desired location.

        :param desired_com: (3,) Desired COM position relative to contact center.
        :return: (nq-7,) Computed joint torques.
        """
        relative_desired_com = desired_com.copy()
        relative_desired_com[2] += TARGET_Z_ADJUSTMENT

        jac_left = self.get_jacobian(self.left_site)
        jac_right = self.get_jacobian(self.right_site)

        # Estimate orientation error and desired angular acceleration
        q = self.data.xquat[self.left_foot_id]
        foot_rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy quaternion in the form [x, y, z, w]
        rotvec = foot_rotation.as_rotvec()
        desired_angular_accel = 100 * rotvec

        # Estimate COM velocity
        base_vel = self.estimate_base_velocity(jac_left)
        com_vel = exponential_filter(self.prev_vel, base_vel, 0.9)
        self.prev_vel = com_vel

        # Compute desired linear acceleration in floor frame
        floor_frame = self.construct_frame_from_foot()
        relative_com = floor_frame @ (self.com - self.get_contact_center())
        desired_linear_accel = 10 * (relative_desired_com - relative_com) - com_vel

        # Compute desired contact forces and map to joint torques
        contact_forces = self.compute_desired_contact_forces(floor_frame, desired_linear_accel, desired_angular_accel)
        left_force = np.concatenate((floor_frame.T @ contact_forces[:3, 0], floor_frame.T @ contact_forces[3:6, 0]))
        right_force = np.concatenate((floor_frame.T @ contact_forces[6:9, 0], floor_frame.T @ contact_forces[9:, 0]))
        joint_torques = -(jac_left.T @ left_force + jac_right.T @ right_force)
        return joint_torques[6:]



