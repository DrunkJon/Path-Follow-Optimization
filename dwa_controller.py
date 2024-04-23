from types import NoneType
from environment import Environment
from typing import Tuple
import numpy as np
from Turtlebot_Kinematics import move_turtle, translate_differential_drive
import shapely
from functions import sigmoid


class DWA_Controller:

    dist_koeff = -500
    heading_koeff = 10
    speed_koeff = 1
    comfort_dist = 2

    def __init__(self, samples=20, max_v=22.2, min_v=-22.2, horizon=10) -> None:
        self.samples = samples
        self.max_v = max_v
        self.min_v = min_v
        self.horizon = horizon

    def __call__(self, env: Environment, dt=0.05):
        v_space = np.linspace(self.min_v, self.max_v, self.samples)
        sensor_fusion = env.get_sensor_fusion()
        # grid sampling
        best_fit = -np.inf
        best_v = None
        for v_right in v_space:
            for v_left in v_space:
                v, w = translate_differential_drive(v_left, v_right)
                # print(v_left, v_right, "->", v_trans)
                fit = self.fitness(env, env.get_internal_state(), v, w, dt*self.horizon, sensor_fusion=sensor_fusion)
                if fit > best_fit:
                    best_fit = fit
                    best_v = (v,w)
        if type(best_v) == NoneType:
            print("ERROR could not find best velocity", best_fit)
        else:
            print("best:", best_v, best_fit)
            return best_v
        
    def fitness(self, env: Environment, cur_state: np.ndarray, v:float, w:float, dt, sensor_fusion=None):
        if type(sensor_fusion) == NoneType:
            sensor_fusion = env.get_sensor_fusion()
        next_state = move_turtle(cur_state, v, w, dt)

        if not sensor_fusion.is_empty:
            pos_point = shapely.Point(next_state[:2])
            dist = (pos_point.distance(sensor_fusion) / env.robo_radius) 
            if dist <= 1:
                return - np.inf
            dist_fit = (1 - sigmoid((dist - self.comfort_dist / 2) * 4 / self.comfort_dist)) * self.dist_koeff
        else:
            dist_fit = 0

        goal_vec = env.goal_pos - next_state[:2]
        heading_vec = move_turtle(next_state, 10, 0, 1) - next_state
        #print("vecs:", goal_vec, heading_vec)
        heading_fit = (goal_vec @ heading_vec[:2] / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec)) * self.heading_koeff * (np.sign(v) if v != 0 else 1)

        speed_fit = (v / self.max_v) * self.speed_koeff

        #print(f"({v}, {w}):\n{dist_fit}\n{heading_fit}\n{speed_fit}")

        return dist_fit + heading_fit + speed_fit
        

        