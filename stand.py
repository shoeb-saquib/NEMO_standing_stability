import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_stabilizer.mujoco_utils import MujocoUtils
from stabilizer.stabilizer import Stabilizer

BASE_COM = np.array([-0.04, 0, 0.4])
COM_OFFSET = np.array([0.0, 0.0, 0.1])
DESIRED_COM = BASE_COM + COM_OFFSET
SIT_COM = BASE_COM + np.array([0.0, 0.0, -0.2])
STAND_TIME = 0

def lin_interp(t, t_total, start, end):
    return start + (t / t_total) * (end - start)

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/flat_scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    stabilizer = Stabilizer()
    dt = model.opt.timestep
    t = 0
    # data.qpos = 0.0
    # data.qpos[2] = 0.68
    data.qpos = model.keyframe("home").qpos
    mj.mj_step(model, data)
    start_com = None
    time_sum = 0
    time_count = 0
    sit = True
    target_com = DESIRED_COM
    while viewer2.is_running():
        start_time = time.time()
        if add_noise: MujocoUtils.add_random_vels(t, dt, data, noise_std, noise_frequency)
        stabilizer.update_simulation(data.qpos[7:], data.qvel[6:])
        # if start_com is None:
        #     start_com = stabilizer.get_relative_com()
        # if t < STAND_TIME:
        #     target_com = lin_interp(t, STAND_TIME, start_com, DESIRED_COM)
        # else: target_com = DESIRED_COM
        if stabilizer.get_relative_com()[2] > 0.4:
            target_com = SIT_COM
        if stabilizer.get_relative_com()[2] < 0.32:
            target_com = DESIRED_COM
        data.ctrl[:] = stabilizer.calculate_joint_torques(target_com)
        end_time = time.time()
        exec_time = end_time - start_time
        time_sum += exec_time
        time_count += 1
        print(1/(time_sum / time_count), exec_time)
        mj.mj_step(model, data)
        t += dt
        time.sleep(dt)
        viewer2.sync()


if __name__ == "__main__":
    add_noise = True
    noise_std = 0.1
    noise_frequency = 1
    simulate()