import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_ankle_controller import SimpleAnkleController

def get_pos(model, qpos, name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    return qpos[model.jnt_qposadr[joint_id]]

def get_vel(model, data, name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    return data.qvel[model.jnt_dofadr[joint_id]]

def move_actuator_to_pos(model, data, name, pos):
    curr_pos = get_pos(model, data.qpos, name)
    vel = get_vel(model, data, name)
    motor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    torque = 100 * (pos - curr_pos) - 10 * vel
    data.ctrl[motor_id] = torque

def move_all_to_pos(model, data, names, qpos):
    for name in names:
        move_actuator_to_pos(model, data, name, get_pos(model, qpos, name))

def body_pos_xy(model, data, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[bid]
    return np.array([pos[0], pos[1]])

def add_random_vels(t, dt, model, data, noise_std, interval):
    cycles = interval // dt
    if t//dt % cycles == 0:
        print(t, ": ADDED NOISE")
        data.qvel += np.random.normal(0, noise_std, size=data.qvel.shape)

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    dt = model.opt.timestep
    t = -1
    data.qpos = model.keyframe('home').qpos
    stabilizer = SimpleAnkleController(model, data)
    while True:
        add_random_vels(t, dt, model, data, 0.1, 5)
        l_foot_center = body_pos_xy(model, data, "l_foot_pitch")
        r_foot_center = body_pos_xy(model, data, "r_foot_pitch")
        robot_center = [l_foot_center[0], (r_foot_center[1] + l_foot_center[1]) / 2]
        move_all_to_pos(model, data, actuator_names, model.keyframe('home').qpos)
        desired_com = np.array([robot_center[0] + 0.18, robot_center[1], 0])
        stabilizer.step(desired_com, robot_center)
        mj.mj_step(model, data)
        stabilizer.show_debug_markers(desired_com)
        t += dt
        time.sleep(dt)
        viewer2.sync()

if __name__ == "__main__":
    simulate()

