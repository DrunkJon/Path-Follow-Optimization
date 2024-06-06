from environment import Environment
from typing import Tuple
import numpy as np
from Turtlebot_Kinematics import KinematicModel, unicycleKin
import shapely
from functions import sigmoid
from controller import Controller


class DWA_Controller(Controller):

    dist_koeff = -500
    heading_koeff = 15
    speed_koeff = 5
    comfort_dist = 3.0

    def __init__(self, samples=20, kinematic: KinematicModel = None, virtual_dt = 2.0) -> None:
        self.samples = samples
        self.kinematic = kinematic if not kinematic is None else unicycleKin()
        self.virtual_dt = virtual_dt

    def __call__(self, env: Environment, dt=2.0, sensor_fusion = None):
        if sensor_fusion is None:
            sensor_fusion == env.get_sensor_fusion()
        # grid sampling
        best_fit = -np.inf
        best_v = None
        for v1, v2 in self.kinematic.v_gen(self.samples):
            fit = self.fitness(env, env.get_internal_state(), v1, v2, self.virtual_dt, sensor_fusion=sensor_fusion)
            if fit > best_fit:
                best_fit = fit
                best_v = (v1,v2)
        if best_v is None:
            print("ERROR could not find best velocity", best_fit)
        else:
            print("best:", best_v, best_fit)
            return best_v
        
    def fitness(self, env: Environment, cur_state: np.ndarray, v1:float, v2:float, dt, sensor_fusion=None):
        if sensor_fusion is None:
            sensor_fusion = env.get_sensor_fusion()
        next_state = self.kinematic(cur_state, v1, v2, dt)

        if not sensor_fusion.is_empty:
            dist_fit = np.inf
            for state in [self.kinematic(cur_state, v1, v2, dt / 5 * i) for i in range(5)]:
                pos_point = shapely.Point(state[:2])
                dist = (pos_point.distance(sensor_fusion) / env.robo_radius) 
                if dist <= 1:
                    return - np.inf
                dist_fit = min(dist_fit, (1 - sigmoid((dist - self.comfort_dist / 2) * 4 / self.comfort_dist)) * self.dist_koeff)
        else:
            dist_fit = 0

        goal_vec = env.goal_pos - next_state[:2]
        heading_vec = self.kinematic.heading(next_state, v1, v2)
        #print("vecs:", goal_vec, heading_vec)
        heading_fit = (goal_vec @ heading_vec[:2] / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec)) * self.heading_koeff * (np.sign(v1) if v1 != 0 else 1)

        speed_fit = self.kinematic.relativ_speed(v1, v2) * self.speed_koeff

        #print(f"({v}, {w}):\n{dist_fit}\n{heading_fit}\n{speed_fit}")

        return dist_fit + heading_fit + speed_fit
        

        