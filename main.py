import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_ankle_controller import SimpleAnkleController
from mujoco_utils import MujocoUtils

add_noise = False
noise_std = 0.1
noise_frequency = 2

sitting_time = 2
time_to_stand = 3

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    dt = model.opt.timestep
    t = 0
    stabilizer = SimpleAnkleController(model, data)
    sit_pos = model.keyframe('sit').qpos
    stand_pos = model.keyframe('home').qpos
    data.qpos = sit_pos
    start_desired_com_offset = 0.55
    end_desired_com_offset = 0.18
    while True:
        l_foot_center = MujocoUtils.body_pos_xy(model, data, "l_foot_pitch")
        r_foot_center = MujocoUtils.body_pos_xy(model, data, "r_foot_pitch")
        robot_center = [l_foot_center[0], (r_foot_center[1] + l_foot_center[1]) / 2]
        if t < sitting_time:
            MujocoUtils.move_all_to_pos(model, data, actuator_names, sit_pos)
            desired_com_offset = start_desired_com_offset
        elif t < time_to_stand + sitting_time:
            new_pos = np.concatenate([data.qpos[:7], MujocoUtils.lin_interp(t - sitting_time, time_to_stand, sit_pos[7:], stand_pos[7:])])
            MujocoUtils.move_all_to_pos(model, data, actuator_names, new_pos)
            desired_com_offset = MujocoUtils.lin_interp(t - sitting_time, time_to_stand, start_desired_com_offset, end_desired_com_offset)
        else:
            MujocoUtils.move_all_to_pos(model, data, actuator_names, stand_pos)
            desired_com_offset = end_desired_com_offset
        if add_noise: MujocoUtils.add_random_vels(t, dt, model, data, 0.08, 2)
        desired_com = np.array([robot_center[0] + desired_com_offset, robot_center[1], 0])
        stabilizer.step(desired_com, robot_center)
        mj.mj_step(model, data)
        stabilizer.show_debug_markers(desired_com)
        t += dt
        time.sleep(dt)
        viewer2.sync()

if __name__ == "__main__":
    simulate()

