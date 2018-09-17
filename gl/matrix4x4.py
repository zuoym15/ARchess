import numpy as np


def translate(v):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = np.squeeze(v)
    return m


def scale(v):
    m = np.eye(4, dtype=np.float32)
    for i in range(3):
        m[i, i] = v[i]

    return m

def flip():
    m = np.eye(4, dtype=np.float32)
    m[1,1] = -1

    return m


def rodrigues(r):
    r = np.array(r, dtype=np.float32)
    theta = np.linalg.norm(r)
    r /= theta

    R = np.eye(4)
    R[:3, :3] = np.cos(theta) * np.eye(3) \
                + (1 - np.cos(theta)) * np.outer(r, r) \
                + np.sin(theta) * np.array([[0, -r[2], r[1]],
                                            [r[2], 0, -r[0]],
                                            [-r[1], r[0], 0]
                                            ])
    return R.astype(np.float32)


def perspective(fov_deg, aspect, z_near, z_far):
    assert (aspect != 0.0)
    assert (z_near != z_far)

    fov = np.radians(fov_deg)
    tan_half_fov = np.tan(fov / 2.0)

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * tan_half_fov)
    m[1, 1] = 1.0 / tan_half_fov
    m[2, 2] = -(z_far + z_near) / (z_far - z_near)
    m[2, 3] = -(2.0 * z_far * z_near) / (z_far - z_near)
    m[3, 2] = -1.0
    return m


def orthographic(r, t, z_near, z_far):
    m = np.array([[1 / r, 0, 0, 0],
                  [0, 1 / t, 0, 0],
                  [0, 0, -2 / (z_far - z_near), -(z_far + z_near) / (z_far - z_near)],
                  [0, 0, 0, 1]], dtype=np.float32)

    return m


def intrinsics_to_perspective(focal, principal, z_near, z_far, width, height):
    fx, fy = focal
    cx, cy = principal

    m = np.array([[2 * fx / width, 0, 2 * cx / width - 1, 0],
                  [0, -2 * fy / height, 1 - 2 * cy / height, 0],
                  [0, 0, (z_far + z_near) / (z_far - z_near), -(2.0 * z_far * z_near) / (z_far - z_near)],
                  [0, 0, 1, 0]], dtype=np.float32)
    return m
