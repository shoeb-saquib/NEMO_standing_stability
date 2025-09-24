import mujoco as mj
import numpy as np

class SimpleAnkleController:

    def __init__(self, model, data):
        self.model = model
        self.data = data

