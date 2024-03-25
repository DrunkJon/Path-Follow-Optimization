from types import NoneType
from environment import Environment
from typing import Tuple
import numpy as np
from Turtlebot_Kinematics import move_turtle, translate_differential_drive


class DWA_Controller:

    def __init__(self, samples=20, max_v=22.2, min_v=-22.2) -> None:
        self.samples = samples
        self.max_v = max_v
        self.min_v = min_v

    def __call__(self, env: Environment, dt=0.05):
        dt = 1 / 20
        v_space = np.linspace(self.min_v, self.max_v, self.samples)
        sensor_fusion = env.get_sensor_fusion()
        # grid sampling
        best_fit = np.inf
        best_v = None
        for v_right in v_space:
            for v_left in v_space:
                v_trans = translate_differential_drive(v_left, v_right)
                fit = env.fitness_single(pos=move_turtle(env.get_internal_state() ,*v_trans, dt)[:2], sensor_fusion=sensor_fusion)
                if fit < best_fit:
                    best_fit = fit
                    best_v = v_trans
        if type(best_v) == NoneType:
            print("ERROR could not find best velocity", best_fit)
        else:
            return best_v
        