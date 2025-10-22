import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_stabilizer import SimpleStabilizer
from mujoco_utils import MujocoUtils

add_noise = True
noise_std = 0.1
noise_frequency = 2

sitting_time = 2
time_to_stand = 3

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
    data_real = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)

    actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]
    actuators = dict(zip(actuator_names, actuator_ids))

    dt = model.opt.timestep
    t = 0

    stabilizer = SimpleStabilizer(model, data_real)

    sit_pos = model.keyframe('sit').qpos
    stand_pos = model.keyframe('stand').qpos

    start_desired_com_offset = 0.55
    end_desired_com_offset = 0.22

    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")

    data.qpos = model.keyframe('home').qpos
    data_real.qpos = model.keyframe('home').qpos

    num_fallen = 0
    time_total = 0
    while True:
        data_real.qpos = model.keyframe('home').qpos
        data_real.qpos[7:] = data.qpos[7:]

        if add_noise: MujocoUtils.add_random_vels(t, dt, data, noise_std, noise_frequency)

        mj.mj_step(model, data_real)
        if t < sitting_time:
            for name in actuators:
                data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, sit_pos, name)
            desired_com_offset = start_desired_com_offset
        elif t < time_to_stand + sitting_time:
            new_pos = np.concatenate([data.qpos[:7], MujocoUtils.lin_interp(t - sitting_time, time_to_stand, sit_pos[7:], stand_pos[7:])])
            for name in actuators:
                data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, new_pos, name)
            desired_com_offset = MujocoUtils.lin_interp(t - sitting_time, time_to_stand, start_desired_com_offset, end_desired_com_offset)
        else:
            for name in actuators:
                data.ctrl[actuators[name]] = MujocoUtils.get_pos(model, stand_pos, name)
            desired_com_offset = end_desired_com_offset

        stabilizer.step(desired_com_offset, data.sensordata)
        data.ctrl[actuators["l_foot_pitch"]] = stabilizer.pitch_torque
        data.ctrl[actuators["r_foot_pitch"]] = stabilizer.pitch_torque
        data.ctrl[actuators["l_foot_roll"]] = stabilizer.roll_torque
        data.ctrl[actuators["r_foot_roll"]] = stabilizer.roll_torque
        if  t >= time_to_stand + sitting_time:
            data.ctrl[actuators["l_hip_yaw"]] = stabilizer.hip_yaw_pos
            data.ctrl[actuators["r_hip_yaw"]] = stabilizer.hip_yaw_pos

        if data.xpos[bid][2] < 0.3:
            reset_robot(model, data)
            time_total += t
            t = 0
            num_fallen += 1

        mj.mj_step(model, data)
        t += dt
        time.sleep(dt)
        viewer2.sync()

if __name__ == "__main__":
    simulate()

