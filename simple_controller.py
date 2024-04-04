from environment import Environment
import numpy as np

def controll(env: Environment, max_v = 75, max_w = np.pi / 3):
    scans = env.get_distance_scans()
    min_dir = np.argmin(scans)
    dist = scans[min_dir]
    min_dir = min_dir if min_dir < 180 else min_dir - 360
    min_deg = np.pi / 180 * min_dir
    target_dist = 50
    b = 5
    attraction = 1
    c = target_dist**2 / np.log(b/attraction)
    repulsion = b * np.exp(- dist**2 / c)
    rel_min_deg = (1 - abs(min_deg) / (np.pi * 0.25))
    w = - max_w * repulsion / 5 * np.sign(min_deg) if abs(min_deg) < 90 else 0
    v = max_v - repulsion / 5 * max_v
    return v,w
    
    
    