import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_stabilizer import SimpleStabilizer
from mujoco_utils import MujocoUtils

add_noise = False
noise_std = 0.05
noise_frequency = 2

sitting_time = 3
time_to_stand = 2

def reset_robot(model, data):
    data.qpos = model.keyframe('sit').qpos
    data.qvel = model.keyframe('sit').qvel
    data.act[:] = 0
    data.ctrl[:] = 0
    data.qacc[:] = 0
    data.xfrc_applied[:] = 0

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/flat_scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)

    actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]
    actuators = dict(zip(actuator_names, actuator_ids))

    dt = model.opt.timestep
    t = 0

    stabilizer = SimpleStabilizer()

    sit_pos = np.array([
        -0.9673, 0, 0, 2.13, -1.0735, 0,
        -0.9673, 0, 0, 2.13, -1.0735, 0,
        0, 0.05, 0, 0, -0.05, 0
    ])
    stand_pos = np.array([
        -1.06169705, 0, 0, 1.22173, -0.08, 0,
        -1.06169705, 0, 0, 1.22173, -0.08, 0,
        0, 0.05, 0, 0, -0.05, 0
    ])

    start_desired_com_offset = 0.55
    end_desired_com_offset = 0.22

    data.qpos = 0
    data.qpos[2] = 0.68

    while True:
        if add_noise: MujocoUtils.add_random_vels(t, dt, data, noise_std, noise_frequency)
        if t < sitting_time:
            for name in actuators:
                data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, np.concatenate([data.qpos[:7], sit_pos]), name)
            desired_com_offset = start_desired_com_offset
        elif t < time_to_stand + sitting_time:
            new_pos = np.concatenate([data.qpos[:7], MujocoUtils.lin_interp(t - sitting_time, time_to_stand, sit_pos, stand_pos)])
            for name in actuators:
                data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, new_pos, name)
            desired_com_offset = MujocoUtils.lin_interp(t - sitting_time, time_to_stand, start_desired_com_offset, end_desired_com_offset)
        else:
            for name in actuators:
                data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, np.concatenate([data.qpos[:7], stand_pos]), name)
            desired_com_offset = end_desired_com_offset

        stabilizer.step(data.qpos[7:], data.sensordata, desired_com_offset)
        data.ctrl[actuators["l_foot_pitch"]] = stabilizer.pitch_torque
        data.ctrl[actuators["r_foot_pitch"]] = stabilizer.pitch_torque
        data.ctrl[actuators["l_foot_roll"]] = stabilizer.roll_torque
        data.ctrl[actuators["r_foot_roll"]] = stabilizer.roll_torque
        if  t >= 5:
            data.ctrl[actuators["l_hip_yaw"]] = stabilizer.hip_yaw_pos
            data.ctrl[actuators["r_hip_yaw"]] = stabilizer.hip_yaw_pos

        mj.mj_step(model, data)
        t += dt
        time.sleep(dt)
        viewer2.sync()

if __name__ == "__main__":
    simulate()

