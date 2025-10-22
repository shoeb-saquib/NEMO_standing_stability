"""Constants for nemo T1."""

from etils import epath
import mujoco

FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_hand",
    "right_hand",
]

LEFT_FEET_GEOMS = [f"left_foot_{i}" for i in range(1, 5)]
RIGHT_FEET_GEOMS = [f"right_foot_{i}" for i in range(1, 5)]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "pelvis"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

model = mujoco.MjModel.from_xml_path("models/nemo/scene.xml")

left_col = [
    model.geom("left_foot_1").id,
    model.geom("left_foot_2").id,
    model.geom("left_foot_3").id,
    model.geom("left_foot_4").id
]

right_col = [
    model.geom("right_foot_1").id,
    model.geom("right_foot_2").id,
    model.geom("right_foot_3").id,
    model.geom("right_foot_4").id
]

floor_col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

ids = {
    "col": {
        "left_foot": left_col,
        "right_foot": right_col,
        "floor": floor_col
    },
    "ctrl_num": model.nu,
}