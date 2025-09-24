import mujoco as mj
from mujoco import viewer
import numpy as np
import time

def get_pos(model, qpos, name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    return qpos[model.jnt_qposadr[joint_id]]

def get_vel(model, data, name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    return data.qvel[model.jnt_dofadr[joint_id]]

def move_block(model, data, name, pos):
    curr_pos = get_pos(model, data.qpos, name)
    motor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    torque = 100 * (pos - curr_pos)
    data.ctrl[motor_id] = torque

def move_many(model, data, names, qpos):
    for name in names:
        move_block(model, data, name, get_pos(model, qpos, name))

def body_pos_xy(model, data, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[bid]
    return np.array([pos[0], pos[1]])

def site_pos_xy(model, data, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, body_name)
    pos = data.xpos[bid]
    return np.array([pos[0], pos[1]])

def lin_interp(t, t_total, start, end):
    return start + (t / t_total) * (end - start)

def get_com(data):
    return data.subtree_com[0]

def get_gravity_constant(model):
    return abs(model.opt.gravity[2])

def set_ankle_torque(model, data, torque):
    left_ankle_act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "l_foot_pitch")
    right_ankle_act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "r_foot_pitch")
    data.ctrl[left_ankle_act_id] = torque
    data.ctrl[right_ankle_act_id] = torque

def calculate_ankle_torque(model, data, m, g, pz_req, body_name):
    return 0.5*m*g*(pz_req - body_pos_xy(model, data, body_name)[0])

def apply_ankle_torque(model, data, m, g, pz_req):
    set_ankle_torque(model, data, calculate_ankle_torque(model, data, m, g, pz_req, "l_foot_pitch"))

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
    g = get_gravity_constant(model)
    m = np.sum(model.body_mass)
    prev_com = get_com(data).copy()
    com = get_com(data)
    dt = model.opt.timestep
    t = -1
    data.qpos = model.keyframe('home').qpos
    com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "com_marker")
    desired_com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "desired_com_marker")
    pz_req_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pz_req_marker")
    pz_curr_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pz_curr_marker")
    while True:
        add_random_vels(t, dt, model, data, 0.1, 5)
        move_many(model, data, actuator_names, model.keyframe('home').qpos)
        desired_com = site_pos_xy(model, data, "l_foot")[0] + 0.16
        com_vel = (com - prev_com) / dt
        frequency_squared = g / com[2]
        desired_com_acceleration = (-50 * (com[0] - desired_com) - 5 * com_vel[0]) / m
        pz_req = com[0] - desired_com_acceleration / frequency_squared
        apply_ankle_torque(model, data, m, g, pz_req)
        prev_com = com.copy()
        mj.mj_step(model, data)
        data.site_xpos[com_marker] = np.array([com[0], com[1], com[2]])
        data.site_xpos[desired_com_marker] = np.array([desired_com, com[1], com[2]])
        data.site_xpos[pz_req_marker] = np.array([pz_req, com[1], 0])
        data.site_xpos[pz_curr_marker] = np.array([body_pos_xy(model, data, "l_foot_pitch")[0], com[1], 0])
        t += dt
        time.sleep(dt)
        viewer2.sync()

if __name__ == "__main__":
    simulate()

