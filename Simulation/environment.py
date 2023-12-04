import numpy as np
from typing import List
import pygame
from Turtlebot_Kinematics import rotate, move_turtle
import json
import time


def array_to_vec(vec: np.ndarray) -> pygame.Vector2:
    return pygame.Vector2(vec[0], vec[1])

def vec_angle(v: np.ndarray, u: np.ndarray) -> float:
    return np.arccos((v @ u) / (np.linalg.norm(v) * np.linalg.norm(u)))


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

    def render(self, surface: pygame.Surface):
        if len(self.corners) >= 2:
            points = [array_to_vec(a) for a in self.translate_corners()]
            if self.finished:
                pygame.draw.polygon(surface, "grey", points)
            pygame.draw.aalines(surface, "black", self.finished, points)
        else:
            for corner in self.translate_corners():
                pygame.draw.circle(surface, "black", array_to_vec(corner), 5)

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
    obstacles: List[Obstacle]
    cur_ob: Obstacle
    robo_state: np.ndarray
    goal_pos: np.ndarray

    def __init__(self, robo_state:np.ndarray, goal_pos:np.ndarray, record = False) -> None:
        self.obstacles = []
        self.cur_ob = None
        self.robo_state = robo_state
        self.goal_pos = goal_pos
        self.record = record
        if record:
            self.robo_record = [tuple(self.robo_state)]
            self.goal_record = [tuple(self.goal_pos)]
        
    def step(self, v,w,dt, surface=None):
        self.update_robo_state(v,w,dt)
        if surface != None:
            self.render(surface)
        self.record_state()
        
    def finish_up(self, data_path = None):
        if data_path == None:
            data_path = f"./data/{'_'.join(map(str,time.localtime()))}.json"
        data = {
            "robo_state": self.robo_record,
            "goal_pos": self.goal_record
        }
        # TODO: use pandas
        # Feather HDL5, Parquet, SQL (saves datatype)
        with open(data_path, "w") as file:
            json.dump(data, file)
            
    def record_state(self):
        if self.record:
            self.robo_record.append(tuple(self.robo_state))
            self.goal_record.append(tuple(self.goal_pos))

    def render(self, surface: pygame.Surface):
        for ob in self.obstacles:
            ob.render(surface)
        if self.cur_ob != None:
            self.cur_ob.render(surface)
        self.render_goal(surface)
        self.render_robo(surface)

    def render_robo(self, surface: pygame.Surface):
        robo_vec = array_to_vec(self.robo_state)
        pygame.draw.circle(surface, "black", robo_vec, 15)
        direction_delta = rotate(np.array([15,0]), self.robo_state[2])
        line_end = pygame.Vector2(robo_vec.x + direction_delta[0], robo_vec.y + direction_delta[1])
        pygame.draw.aaline(surface, "white", robo_vec, line_end)

    def render_goal(self, surface: pygame.Surface):
        pygame.draw.circle(surface, "yellow", array_to_vec(self.goal_pos), 10)
        pygame.draw.circle(surface, "black", array_to_vec(self.goal_pos), 10, 2)

    def turn_robo_towards(self, point: np.ndarray):
        if (point != self.get_robo_pos()).any():
            delta = point - self.get_robo_pos()
            angle = vec_angle(np.array([1,0]), delta)
            self.robo_state[2] = angle if delta[1] <= 0 else np.pi * 2 -angle

    def update_robo_state(self, v, w, dt):
        self.robo_state = move_turtle(self.robo_state, v, w, dt)

    def get_robo_angle(self) -> float:
        return self.robo_state[2]
    
    def get_robo_pos(self) -> np.ndarray:
        return self.robo_state[:2]

    def set_robo_state(self, state:np.ndarray):
        self.robo_state = state

    def set_goal_pos(self, pos: np.ndarray):
        self.goal_pos = pos

    def get_distance_scans(self, render_surface: pygame.Surface = None):
        robo_pos = self.robo_state[:2]
        robo_angle = self.robo_state[2]
        x_axis = np.array([1,0])
        angles = np.linspace(robo_angle, np.pi * 2 + robo_angle, 360)
        directions = [rotate(x_axis, angle) for angle in angles]
        distances = [self.scan(robo_pos, direction) for direction in directions]
        if render_surface != None:
            for i, (dist, dire) in enumerate(zip(distances, directions)):
                if i % 3 != 0: continue
                cords = robo_pos + min(dist, 500) * dire
                pygame.draw.aaline(render_surface, "red" if i != 0 else "blue", array_to_vec(robo_pos), array_to_vec(cords))
                self.render_robo(render_surface)
        return distances
    
    def scan(self, robo_pos: np.ndarray, direction: np.ndarray):
        return min([ob.scan(robo_pos, direction) for ob in self.obstacles])

    def add_corner(self, corner:np.ndarray):
        if self.cur_ob == None:
            self.cur_ob = Obstacle(corner)
        else:
            self.cur_ob.add_corner(corner)

    def finish_obstacle(self):
        self.cur_ob.finished = True
        self.obstacles.append(self.cur_ob)
        self.cur_ob = None

    def to_json(self):
        data = {
            "robo_state": list(self.robo_state),
            "goal_pos": list(self.goal_pos),
            "obstacles": list(ob.to_dict() for ob in self.obstacles)
        }
        with open(f"./levels/{'_'.join(map(str,time.localtime()))}.json", "w") as file:
            file.write(json.dumps(data, indent=2))

    def from_json(json_string:str, record=False) -> "Environment":
        data = json.loads(json_string)
        new_env = Environment(np.array(data["robo_state"], dtype=float), np.array(data["goal_pos"], dtype=float), record)
        new_env.obstacles = list([Obstacle.from_dict(ob_dict) for ob_dict in data["obstacles"]])
        return new_env
    
