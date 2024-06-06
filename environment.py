import numpy as np
from typing import List
from Turtlebot_Kinematics import KinematicModel, unicycleKin, rotate
import json
import time
import pandas as pd
import shapely
from shapely.ops import nearest_points
from functions import sigmoid, vec_angle
from os import listdir


def load_ENV(filename, kinematic:KinematicModel, record:bool):
    env_files = sorted(listdir("levels"), reverse=True)
    if filename != None and f"{filename}.json" in env_files:
        print(f"loading {filename}")
        with open(f"./levels/{filename}.json", "r") as file:
            json_str = file.read()
            ENV = Environment.from_json(json_str, kinematic, record)
    elif len(env_files) >= 1:
        print(f"loading {env_files[0]}")
        with open(f"./levels/{env_files[0]}", "r") as file:
            json_str = file.read()
            ENV = Environment.from_json(json_str, kinematic, record)
    else:
        print("loading new ENV")
        ENV = Environment(np.array([640,360,0], dtype=float), np.array([1260, 700], dtype=float), kinematic, record)
    return ENV


def random_koeff(max_diff = 0.05):
    return 1 - max_diff + np.random.rand() * 2 * max_diff


class Obstacle:
    corners: List[np.ndarray]
    offset: np.ndarray

    def __init__(self, offset:np.ndarray) -> None:
        self.offset = offset
        self.corners = [np.zeros(2, dtype=float)]
        self.finished = False

    def add_corner(self, corner: np.ndarray):
        self.corners.append(corner - self.offset)

    def translate_corners(self):
        return list([corner + self.offset for corner in self.corners])

    def get_lines(self):
        # returns corner and direction (not normed!) to next corner for each line of obstacle
        trans_corn = self.translate_corners()
        return [(c, c - trans_corn[(i+1)%len(trans_corn)])  for i, c in enumerate(trans_corn)]

    def scan(self, agent_pos: np.ndarray, direction: np.ndarray):
        hits = [np.Infinity]
        lines = self.get_lines()
        for c, c_dir in lines:
            A = np.array([direction, c_dir]).transpose()
            b = c - agent_pos
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as err:
                print(err)
                continue
            if x[0] >= 0 and 0 <= x[1] <= 1:
                hits.append(x[0])
        return min(hits)
    
    def to_dict(self):
        return {
            "offset": list(self.offset),
            "corners": list(list(c) for c in self.corners)
        }

    @staticmethod
    def from_dict(d:dict) -> "Obstacle":
        new_obs = Obstacle(np.array(d["offset"]))
        new_obs.corners = list([np.array(c) for c in d["corners"]])
        new_obs.finished = True
        return new_obs


class Environment:
    map_obstacles: List[Obstacle]
    unknown_obstacles: List[Obstacle]
    cur_ob: Obstacle
    robo_state: np.ndarray
    goal_pos: np.ndarray
    data: pd.DataFrame

    # values for scan and rendering
    max_scan_dist = 500
    robo_radius = 16
    scan_lines = 90
    # values for fitness function
    collision_penalty = 50000
    goal_koeff = 50
    speed_koeff = 3
    obstacle_koeff = 200
    heading_koeff = 0
    comfort_dist = 3.0    # * robo_radius (> 1, sonst ist nur collision relevant)

    def __init__(self, robo_state:np.ndarray, goal_pos:np.ndarray, kinematic: KinematicModel = None, record = False, use_errors=False) -> None:
        self.kinematic = kinematic if not kinematic is None else unicycleKin()
        self.use_erros= use_errors
        self.map_obstacles = []
        self.unknown_obstacles = []
        self.cur_ob = None
        self.robo_state = robo_state
        self.internal_offset = np.zeros_like(self.robo_state)
        # self.internal_offset = np.array([np.random.random() * 50, np.random.random() * 50, np.random.random() * (2*np.pi / 360)*10,])
        self.goal_pos = goal_pos
        self.record = record
        self.time = 0.0
        if record:
            self.data = pd.DataFrame({
                "robo_x" : self.robo_state[0],
                "robo_y" : self.robo_state[1],
                "robo_deg" : self.robo_state[2],
                "goal_x" : self.goal_pos[0],
                "goal_y" : self.goal_pos[1],
                },
                index = [self.time]
            )

    def get_internal_state(self):
        return self.robo_state + self.internal_offset
        
    def step(self, v1, v2,dt):
        self.update_robo_state(v1,v2,dt)
        self.time += dt
        self.record_state()
        
    def finish_up(self, data_path = None):
        if self.record:
            if data_path == None:
                data_path = f"./data/{'_'.join(map(str,time.localtime()))}.h5"
            if not data_path.endswith(".h5"):
                data_path += ".h5"
            self.data.to_hdf(data_path, "data")
            
    def record_state(self):
        if self.record:
            self.data.loc[self.time] = {
                "robo_x" : self.robo_state[0],
                "robo_y" : self.robo_state[1],
                "robo_deg" : self.robo_state[2],
                "goal_x" : self.goal_pos[0],
                "goal_y" : self.goal_pos[1],
            }

    def turn_robo_towards(self, point: np.ndarray):
        if (point != self.get_robo_pos()).any():
            delta = point - self.get_robo_pos()
            angle = vec_angle(np.array([1,0]), delta)
            self.robo_state[2] = angle if delta[1] <= 0 else np.pi * 2 -angle

    def update_robo_state(self, v1, v2, dt):
        old_state = self.robo_state
        self.robo_state = self.kinematic(old_state, v1 * random_koeff(0.05 if self.use_erros else 0), v2 * random_koeff(0.05 if self.use_erros else 0), dt * random_koeff())
        if self.use_erros:
            self.internal_offset = self.kinematic(old_state + self.internal_offset, v1, v2, dt) - self.robo_state

    def get_robo_angle(self) -> float:
        return self.robo_state[2]
    
    def get_robo_pos(self) -> np.ndarray:
        return self.robo_state[:2]

    def set_robo_state(self, state:np.ndarray):
        self.robo_state = state

    def set_goal_pos(self, pos: np.ndarray):
        self.goal_pos = pos

    def get_map_poly(self):
        return shapely.unary_union([shapely.LinearRing(obs.translate_corners()) for obs in self.map_obstacles + self.unknown_obstacles])
    
    def get_dist_to_goal(self, internal = False):
        if internal:
            return np.linalg.norm(self.goal_pos - self.get_internal_state()[:2])
        else:
            return np.linalg.norm(self.goal_pos - self.get_robo_pos())

    def get_scan_coords(self):
        poly = self.get_map_poly()
        robo_pos = self.get_robo_pos()
        robo_point = shapely.Point(robo_pos)
        robo_angle = self.get_robo_angle()
        sens_dist = np.array([self.max_scan_dist,0])
        angles = np.linspace(robo_angle, np.pi * 2 + robo_angle, self.scan_lines)
        directions = [rotate(sens_dist, angle) for angle in angles]
        lines = [shapely.LineString([robo_pos, robo_pos + direc]) for direc in directions]
        intersects = [line.intersection(poly) for line in lines]
        closest_poss = [(robo_pos + directions[i]).flatten() if inter.is_empty else np.array(nearest_points(inter, robo_point)[0].coords).flatten() for i, inter in enumerate(intersects)]
        return closest_poss
    
    def get_distance_scans(self, scan_coords = None):
        if scan_coords == None:
            scan_coords = self.get_scan_coords()
        distances = [np.linalg.norm(s - self.get_robo_pos()) for s in scan_coords]
        # add sens errors if scan dist < max dist
        distances = [d * (random_koeff() if self.use_erros else 1) if d < self.max_scan_dist - 0.1 else d for d in distances]
        return distances
    
    def scan(self, robo_pos: np.ndarray, direction: np.ndarray):
        return min([ob.scan(robo_pos, direction) for ob in self.map_obstacles + self.unknown_obstacles])

    def get_internal_scan_cords(self, scan_dists = None):
        if scan_dists == None:
            scan_dists = self.get_distance_scans()
        x, y, rho = self.get_internal_state()
        internal_pos = np.array([x,y])
        directions = [
            rotate(np.array([1.0,0.0]), angle) 
            for angle in 
            np.linspace(rho, np.pi * 2 + rho, len(scan_dists))
        ]
        return [internal_pos + dist * dire for i, (dist, dire) in enumerate(zip(scan_dists, directions))]

    def get_sensor_fusion(self, point_cloud = True) -> shapely.MultiPolygon:
        scan_dists = self.get_distance_scans()
        scan_cords = self.get_internal_scan_cords(scan_dists)
        scan_point_cloud = shapely.unary_union([shapely.Point(cord) for cord,dist in zip(scan_cords, scan_dists) if dist <= self.max_scan_dist - 0.1])
        if scan_point_cloud.is_empty:
            print("scan empty")
        map_poly = shapely.unary_union([shapely.Polygon(obs.translate_corners()) for obs in self.map_obstacles])
        if point_cloud:
            return scan_point_cloud.union(map_poly.difference(shapely.Polygon(scan_cords)))
        else:
            return scan_point_cloud.buffer(5, quad_segs = 3).union(map_poly.difference(shapely.Polygon(scan_cords)))
        
    def get_obstacle_dist(self, sensor_fusion = None):
        state = self.get_internal_state()
        if sensor_fusion == None:
            sensor_fusion = self.get_sensor_fusion()
        pos_point = shapely.Point(state[:2])
        return pos_point.distance(sensor_fusion) / self.robo_radius

    # minimize:
    def fitness_single(self, state = None, sensor_fusion = None, v: np.array = np.zeros(2)):
        if state is None:
            state = self.get_internal_state()
        if sensor_fusion == None:
            sensor_fusion = self.get_sensor_fusion()
        pos_point = shapely.Point(state[:2])
        obstacle_dist = pos_point.distance(sensor_fusion) / self.robo_radius
        goal_dist = pos_point.distance(shapely.Point(self.goal_pos)) / self.robo_radius
        if obstacle_dist <= 1:
            return self.collision_penalty
        goal_fit = self.goal_koeff * (goal_dist)
        try:
            obstacle_fit = 0 if sensor_fusion.is_empty or obstacle_dist >= self.comfort_dist else self.obstacle_koeff * (self.comfort_dist - obstacle_dist) / self.comfort_dist
        except OverflowError as err:
            print(err, "\n", "obstacle_dist =", obstacle_dist)
            obstacle_fit = self.obstacle_koeff
        goal_vec = self.goal_pos - state[:2]
        heading_vec = self.kinematic.heading(state, v[0], v[1])
        heading_fit = -(goal_vec @ heading_vec[:2] / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec)) * self.heading_koeff
        # return  goal_fit + obstacle_fit # + heading_fit
        return  goal_fit + obstacle_fit - self.speed_koeff * np.linalg.norm(v)

    def add_corner(self, corner:np.ndarray):
        if self.cur_ob == None:
            self.cur_ob = Obstacle(corner)
        else:
            self.cur_ob.add_corner(corner)

    def finish_obstacle(self, add_to_map = True):
        self.cur_ob.finished = True
        if add_to_map:
            self.map_obstacles.append(self.cur_ob)
        else:
            self.unknown_obstacles.append(self.cur_ob)
        self.cur_ob = None

    def to_json(self):
        data = {
            "robo_state": list(self.robo_state),
            "goal_pos": list(self.goal_pos),
            "obstacles": list(ob.to_dict() for ob in self.map_obstacles),
            "unknowns": list(ob.to_dict() for ob in self.unknown_obstacles)
        }
        with open(f"./levels/{'_'.join(map(str,time.localtime()))}.json", "w") as file:
            file.write(json.dumps(data, indent=2))

    def from_json(json_string:str, kinematic:KinematicModel = None, record=False) -> "Environment":
        data = json.loads(json_string)
        new_env = Environment(np.array(data["robo_state"], dtype=float), np.array(data["goal_pos"], dtype=float), kinematic=kinematic, record=record)
        new_env.map_obstacles = list([Obstacle.from_dict(ob_dict) for ob_dict in data["obstacles"]])
        new_env.unknown_obstacles = list([Obstacle.from_dict(ob_dict) for ob_dict in data["unknowns"]])
        return new_env
    
