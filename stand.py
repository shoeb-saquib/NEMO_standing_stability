import mujoco as mj
from mujoco import viewer
import numpy as np
import time
from mujoco_utils import MujocoUtils
from stabilizer import Stabilizer
from omniscient_stabilizer import OmniscientStabilizer
from plotnine import *
import pandas as pd

parameters = ["COM Difference", "COM Offset Difference", "COM Vel Difference", "Desired Linear Acceleration Difference", "Rotation Vector Difference", "Desired Angular Acceleration Difference"]

def display_plot(df, x, y):
    plot = (ggplot(df, aes(x=x, y=y)) + geom_line())
    plot.show()

def simulate():
    model = mj.MjModel.from_xml_path("models/nemo/flat_scene.xml")
    data = mj.MjData(model)
    data_real = mj.MjData(model)
    viewer2 = viewer.launch_passive(model, data)
    stabilizer = Stabilizer(model, data_real)
    true_stabilizer = OmniscientStabilizer(model, data)
    dt = model.opt.timestep
    t = 0
    data.qpos = 0.0
    data.qpos[2] = 0.68
    # data.qpos = model.keyframe('home').qpos
    mj.mj_step(model, data)
    com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "com_marker")
    desired_com_marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "desired_com_marker")
    differences = []
    x = []
    while True:
        if add_noise: MujocoUtils.add_random_vels(t, dt, data, noise_std, noise_frequency)
        data.ctrl[:], com_vel, true_values = true_stabilizer.calculate_joint_torques(dt, desired_offset)
        data.ctrl[:], measured_values = stabilizer.calculate_joint_torques(dt, data.qpos[7:], desired_offset, com_vel)
        # differences.append(list(np.array(measured_values) - np.array(true_values)))
        # x.append(t)
        mj.mj_step(model, data)
        # data.site_xpos[com_marker] = true
        # data.site_xpos[desired_com_marker] = measured
        t += dt
        time.sleep(dt)
        viewer2.sync()
    # dataframe = pd.DataFrame(differences, columns=parameters)
    # dataframe['Time'] = x
    # for parameter in parameters:
    #     display_plot(dataframe, 'Time', parameter)


if __name__ == "__main__":
    add_noise = False
    noise_std = 0.15
    noise_frequency = 2

    home_com = [-4.36481990e-02, -5.25951358e-06, 3.99707890e-01]
    x_offset = 0
    y_offset = 0
    z_offset = -0.25
    desired_offset = np.array([home_com[0] + x_offset, home_com[1] + y_offset, home_com[2] + z_offset])
    simulate()