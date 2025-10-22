import mujoco as mj
import numpy as np

class MujocoUtils:
    @staticmethod
    def get_pos(model, qpos, name):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        return qpos[model.jnt_qposadr[joint_id]]

    @staticmethod
    def get_vel(model, data, name):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        return data.qvel[model.jnt_dofadr[joint_id]]

    @staticmethod
    def lin_interp(t, t_total, start, end):
        return start + (t / t_total) * (end - start)

    @staticmethod
    def move_actuator_to_pos(model, data, name, pos, kp = 110, kd = 5):
        curr_pos = MujocoUtils.get_pos(model, data.qpos, name)
        vel = MujocoUtils.get_vel(model, data, name)
        motor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        torque = kp * (pos - curr_pos) - kd * vel
        data.ctrl[motor_id] = torque

    @staticmethod
    def move_all_to_pos(model, data, names, qpos):
        for name in names:
            MujocoUtils.move_actuator_to_pos(model, data, name, MujocoUtils.get_pos(model, qpos, name))

    @staticmethod
    def body_pos(model, data, body_name):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        return data.xpos[bid]

    @staticmethod
    def add_random_vels(t, dt, data, noise_std, interval):
        cycles = interval // dt
        if int(t // dt) % cycles == 0:
            #print(f"{t:.3f} : ADDED NOISE")
            data.qvel += np.random.normal(0, noise_std, size=data.qvel.shape)
