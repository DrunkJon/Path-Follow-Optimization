from environment import Environment
from typing import Tuple
import numpy as np
from Turtlebot_Kinematics import KinematicModel, unicycleKin
import shapely
from functions import sigmoid
from controller import Controller


class DWA_Controller(Controller):

    dist_koeff = -4
    heading_koeff = 1.3
    speed_koeff = 4
    comfort_dist = 3.0
    crash_dist = 1.2

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
            fit = sum(self.fitness(env, env.get_internal_state(), v1, v2, self.virtual_dt, sensor_fusion=sensor_fusion))
            # print((v1, v2), "->", fit)
            if fit > best_fit:
                best_fit = fit
                best_v = (v1,v2)
        if best_v is None:
            print("ERROR could not find best velocity", best_fit)
            return (-5.0, 0.0)
        else:
            print("best:", best_fit, best_v)
            return best_v
        
    def fitness(self, env: Environment, cur_state: np.ndarray, v1:float, v2:float, dt, sensor_fusion=None):
        if sensor_fusion is None:
            sensor_fusion = env.get_sensor_fusion()
        kin = self.kinematic.clone()
        next_state = kin(cur_state, v1, v2, dt)

        inbetweens = 5
        if not sensor_fusion.is_empty:
            dist_fits = []
            heading_fits = []
            last_state = cur_state
            for i, state in [(i, kin(last_state, v1, v2, dt / inbetweens)) for i in range(1, inbetweens)]:
                pos_point = shapely.Point(state[:2])
                dist = (pos_point.distance(sensor_fusion) / env.robo_radius) 
                if dist <= self.crash_dist:
                    dist_fits.append(-np.inf)
                dist_fits.append(max(1 - (dist / self.comfort_dist), 0) * self.dist_koeff)
                goal_vec = env.get_goal_pos(dt / inbetweens * i) - cur_state[0:2]
                heading_vec = state - cur_state
                if np.linalg.norm(goal_vec) != 0 and np.linalg.norm(heading_vec[:2]) != 0:
                    heading_fits.append(
                        ((goal_vec @ heading_vec[:2]) / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec[:2])) 
                        * self.heading_koeff # * (np.sign(v1) if v1 != 0 else 1)
                    )
                else:
                    heading_fits.append(0)
                last_state = state
            heading_fit = np.average(heading_fits)
            dist_fit = min(dist_fits)
        else:
            dist_fit = 0
            goal_vec = env.goal_pos - next_state[:2]
            heading_vec = kin.heading(next_state, v1, v2)
            heading_fit = (goal_vec @ heading_vec[:2] / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec)) * self.heading_koeff # * (np.sign(v1) if v1 != 0 else 1)

        speed_fit = abs(kin.relativ_speed(v1, v2)) * self.speed_koeff
        assert dist_fit < np.inf
        return dist_fit, heading_fit, speed_fit
        

        