#!/usr/bin/env python
import numpy as np

def rotation_matrix(ang, matrix_type="2d"):
    """
  Roll = 3D rotation around the x axis
  Pitch = 3D rotation around the y axis
  Yaw = 3D rotation around the z axis
  """
    if matrix_type == "2d":
        return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    elif matrix_type in {"x", "roll"}:
        return np.array(
            [[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]
        )
    elif matrix_type in {"y", "pitch"}:
        return np.array(
            [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
        )
    elif matrix_type in {"z", "yaw"}:
        return np.array(
            [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
        )
    else:
        raise Exception("Unrecognized option matrix_type = {}".format(matrix_type))