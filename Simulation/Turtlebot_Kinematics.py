import numpy as np

a = np.array([1,0])
def rotation_matrix(angle):
    return np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])

def rotate(vec: np.ndarray, angle: float):
    return rotation_matrix(angle) @ vec

def move_turtle(state: np.array, v:float, w:float, t:float) -> np.array:
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
