import numpy as np

def exponential_filter(previous, current, alpha):
    return alpha * previous + (1 - alpha) * current

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quat_from_mat(r):
    qw = np.sqrt(1.0 + np.trace(r)) / 2.0
    qx = (r[2, 1] - r[1, 2]) / (4.0 * qw)
    qy = (r[0, 2] - r[2, 0]) / (4.0 * qw)
    qz = (r[1, 0] - r[0, 1]) / (4.0 * qw)
    return np.array([qw, qx, qy, qz])

def skew(v):
    x, y, z = v
    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])