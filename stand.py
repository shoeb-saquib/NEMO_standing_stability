import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from simple_stabilizer.mujoco_utils import MujocoUtils
from stabilizer.stabilizer import Stabilizer
from plotnine import *


def display_plot(df, x, y):
    plot = (ggplot(df, aes(x=x, y=y, color="Category")) + geom_line())
    plot.show()

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/flat_scene.xml")
    data = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    stabilizer = Stabilizer()
    dt = model.opt.timestep
    t = 0
    data.qpos = 0.0
    data.qpos[2] = 0.68
    mj.mj_step(model, data)
    while True:
        if add_noise: MujocoUtils.add_random_vels(t, dt, data, noise_std, noise_frequency)
        data.ctrl[:] = stabilizer.calculate_joint_torques(dt, data.qpos[7:], data.qvel[6:], desired_offset, data.sensordata)
        mj.mj_step(model, data)
        t += dt
        time.sleep(dt)
        viewer2.sync()


if __name__ == "__main__":
    add_noise = True
    noise_std = 0.1
    noise_frequency = 2

    home_com = [-4.36481990e-02, -5.25951358e-06, 3.99707890e-01]
    x_offset = 0
    y_offset = 0
    z_offset = -0.25
    desired_offset = np.array([home_com[0] + x_offset, home_com[1] + y_offset, home_com[2] + z_offset])
    simulate()