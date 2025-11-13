import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_stabilizer.mujoco_utils import MujocoUtils
from stabilizer.stabilizer import Stabilizer

BASE_COM = [-0.04, 0, 0.4]
X_COM_OFFSET = 0.0
Y_COM_OFFSET = 0.05
Z_COM_OFFSET = 0.0

DESIRED_COM = np.array([BASE_COM[0] + X_COM_OFFSET, BASE_COM[1] + Y_COM_OFFSET, BASE_COM[2] + Z_COM_OFFSET])

STAND_TIME = 3


def lin_interp(t, t_total, start, end):
    return start + (t / t_total) * (end - start)

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/flat_scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    stabilizer = Stabilizer()
    dt = model.opt.timestep
    t = -2
    # data.qpos = 0.0
    # data.qpos[2] = 0.68
    data.qpos = model.keyframe("sit").qpos
    mj.mj_step(model, data)
    start_com = None
    while True:
        if add_noise: MujocoUtils.add_random_vels(t, dt, data, noise_std, noise_frequency)
        stabilizer.update_simulation(data.qpos[7:], data.qvel[6:])
        if t < 0:
            data.qpos = model.keyframe("sit").qpos
            data.qvel = np.zeros_like(data.qvel)
        else:
            if start_com is None:
                start_com = stabilizer.get_relative_com()
            if t < STAND_TIME:
                target_com = lin_interp(t, STAND_TIME, start_com, DESIRED_COM)
            else: target_com = DESIRED_COM
            data.ctrl[:] = stabilizer.calculate_joint_torques(target_com)
        mj.mj_step(model, data)
        t += dt
        time.sleep(dt)
        viewer2.sync()


if __name__ == "__main__":
    add_noise = False
    noise_std = 0.1
    noise_frequency = 2

    simulate()