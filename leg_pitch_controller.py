import mujoco as mj
import numpy as np
from mujoco_utils import MujocoUtils


class LegPitchController:
    """
    Computes ankle and knee pitch torques to:
      - keep the torso upright (pitch = 0)
      - maintain a desired torso height
    Similar interface to SimpleAnkleController.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Gains
        self.kp_pitch = 60.0
        self.kd_pitch = 8.0
        self.kp_z = 2000.0
        self.kd_z = 200.0

        # Precompute IDs
        self.imu_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "imu")
        self.ankle_body = "l_foot_pitch"
        self.knee_body = "l_knee"

        # Preallocate Jacobians
        self.Jpos = np.zeros((3, model.nv))
        self.Jrot = np.zeros((3, model.nv))

        # Constants
        self.g = abs(model.opt.gravity[2])
        self.m = np.sum(model.body_mass)

        # Control outputs (accessed by main)
        self.ankle_torque = 0.0
        self.knee_torque = 0.0

    # ---- Helper methods ----

    def _get_up_vector(self):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "upvector")
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + 3].copy()

    def _get_gyro_local(self):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "gyro")
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + 3].copy()

    def _get_imu_height_and_vel(self):
        mj.mj_jacSite(self.model, self.data, self.Jpos, self.Jrot, self.imu_site)
        z = self.data.site_xpos[self.imu_site][2]
        vz = self.Jpos[2, :] @ self.data.qvel
        return z, vz

    def _compute_pitch_error(self):
        up = self._get_up_vector()
        up /= (np.linalg.norm(up) + 1e-9)

        # In x-z plane
        v2 = np.array([up[0], up[2]])
        v2 /= (np.linalg.norm(v2) + 1e-9)
        dot = np.clip(np.dot(v2, np.array([0, 1])), -1.0, 1.0)
        pitch_mag = np.arccos(dot)
        pitch_err = np.sign(up[0]) * pitch_mag

        pitch_rate = self._get_gyro_local()[1]
        return pitch_err, pitch_rate

    # ---- Main control step ----

    def step(self, desired_height):
        """
        Compute torques for ankle and knee to keep torso vertical
        and maintain desired torso (IMU) height.
        """
        # Height control
        z, vz = self._get_imu_height_and_vel()
        pitch_err, pitch_rate = self._compute_pitch_error()

        # Task efforts
        Fz = self.kp_z * (desired_height - z) - self.kd_z * vz
        M_pitch = -self.kp_pitch * pitch_err - self.kd_pitch * pitch_rate

        # Compute Jacobians
        mj.mj_jacSite(self.model, self.data, self.Jpos, self.Jrot, self.imu_site)
        Jz = self.Jpos[2, :]
        Jy_rot = self.Jrot[1, :]

        # Extract joint indices
        ankle_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.ankle_body)
        knee_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.knee_body)
        ankle_dof = self.model.jnt_dofadr[ankle_jid]
        knee_dof = self.model.jnt_dofadr[knee_jid]

        # Combine contributions from both tasks (height & pitch)
        tau_ankle = Jz[ankle_dof] * Fz + Jy_rot[ankle_dof] * M_pitch
        tau_knee  = Jz[knee_dof]  * Fz + Jy_rot[knee_dof]  * M_pitch

        # Update class attributes
        self.ankle_torque = tau_ankle
        self.knee_torque = tau_knee
