import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from leg_pitch_controller import LegPitchController
from mujoco_utils import MujocoUtils

add_noise = False
noise_std = 0.05
noise_frequency = 2

sitting_time = 2
time_to_stand = 3

def keep_body_vertical(model, data, actuators):
    sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer")
    start = model.sensor_adr[sensor_id]
    accel = data.sensordata[start:start + 3]
    accel = np.array(accel) / np.linalg.norm(accel)
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))
    hip_pos = MujocoUtils.get_pos(model, data.qpos, "l_hip_pitch") - pitch
    data.ctrl[actuators["l_hip_pitch"]] = hip_pos
    data.ctrl[actuators["r_hip_pitch"]] = hip_pos

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/flat_scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]
    actuators = dict(zip(actuator_names, actuator_ids))
    dt = model.opt.timestep
    t = 0
    stabilizer = LegPitchController(model, data)
    stand_pos = model.keyframe('home').qpos
    data.qpos = stand_pos
    l_pitch_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "l_foot_pitch")
    r_pitch_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "r_foot_pitch")
    l_knee_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "l_knee")
    r_knee_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "r_knee")
    while True:
        for name in actuators:
            data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, stand_pos, name)
        stabilizer.step(100)
        data.ctrl[l_pitch_act] = stabilizer.ankle_torque
        data.ctrl[r_pitch_act] = stabilizer.ankle_torque
        data.ctrl[l_knee_act] = stabilizer.knee_torque
        data.ctrl[r_knee_act] = stabilizer.knee_torque

        mj.mj_step(model, data)
        t += dt
        time.sleep(dt)
        viewer2.sync()

if __name__ == "__main__":
    simulate()

