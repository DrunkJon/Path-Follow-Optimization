import numpy as np


def sigmoid(x):
    return np.e**x / (1 + np.e**x)

def vec_angle(v: np.ndarray, u: np.ndarray) -> float:
    return np.arccos((v @ u) / (np.linalg.norm(v) * np.linalg.norm(u)))

def function_3(x1:np.ndarray, x2:np.ndarray, d, b=1.1, k=1.8):
    a = k
    b = k * b
    c = d**2 / np.log(b/a)
    return (a - (b * np.exp(- np.linalg.norm(x1-x2)**2 / c))) * (x2 - x1)

def comf_distance(x1:np.ndarray, x2:np.ndarray, d, k=0.5):
    return k * (np.linalg.norm(x2 - x1) - d) * (x2 - x1)