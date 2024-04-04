from simple_controller import controll
import numpy as np
from run_util import *

map_file = "simple"
ENV = load_ENV(map_file, record=True)


target_fps = 20
dt = 1 / target_fps

max_t = 120
end_goal_dist = 0.5 

while ENV.time <= max_t:

    ENV.get_distance_scans()

    v,w = controll(ENV)
    ENV.step(v, w, dt)
    if np.linalg.norm(ENV.get_robo_pos() - ENV.goal_pos) <= end_goal_dist:
        break

ENV.finish_up()