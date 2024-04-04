import numpy as np

a = np.array([1,0])
def rotation_matrix(angle):
    return np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])

def rotate(vec: np.ndarray, angle: float):
    return rotation_matrix(angle) @ vec

def move_turtle(state: np.array, v:float, w:float, t:float) -> np.array:
    if len(state) < 3:
        print(state)
    x, y, theta = state
    if w != 0:
        r = v / w
        wt = w * t
        return np.array([
            x + r * np.sin(wt) * np.cos(theta) + (r - r * np.cos(wt)) * np.sin(theta),
            y + r * np.sin(wt)*(-np.sin(theta))+ (r - r * np.cos(wt)) * np.cos(theta),
            theta + wt
        ])
    else:
        return np.array([
            x + np.cos(theta) * v * t,
            y - np.sin(theta) * v * t,
            theta
        ])
    
def translate_differential_drive(v_left: float, v_right: float, wheel_distance:float = 16.0):
    # based on Kinematic model described by Nele Traichel in her Master Thesis
    trans = (v_right + v_left) / 2
    rotation = (v_right - v_left) / wheel_distance
    # returns translational and rotational speed of robot that results from the input velocities
    return trans, rotation
