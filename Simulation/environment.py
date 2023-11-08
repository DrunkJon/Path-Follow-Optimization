import numpy as np
from typing import List
import pygame
from Turtlebot_Kinematics import rotate


def array_to_vec(vec: np.array) -> pygame.Vector2:
    return pygame.Vector2(vec[0], vec[1])


class Obstacle:
    corners: List[np.ndarray]

    def __init__(self) -> None:
        self.corners = []
        self.finished = False

    def render(self, surface: pygame.Surface):
        pygame.draw.polygon(surface, "black", [array_to_vec(a) for a in self.corners])

    def get_lines(self):
        # returns corner and direction (not normed!) to next corner for each line of obstacle
        return [(c, c - self.corners[(i+1)%len(self.corners)])  for i, c in enumerate(self.corners)]

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


class Environment:
    obstacles: List[Obstacle]
    cur_ob: Obstacle

    def __init__(self) -> None:
        self.obstacles = []
        self.cur_ob = None

    def render(self, surface: pygame.Surface):
        for ob in self.obstacles:
            ob.render(surface)

    def get_distance_scans(self, agent_pos: np.ndarray, agent_angle: float, render_surface: pygame.Surface = None):
        x_axis = np.array([1,0])
        angles = np.linspace(agent_angle, np.pi * 2 + agent_angle, 360)
        directions = [rotate(x_axis, angle) for angle in angles]
        distances = [self.scan(agent_pos, direction) for direction in directions]
        if render_surface != None:
            for i, (dist, dire) in enumerate(zip(distances, directions)):
                if i % 3 != 0: continue
                cords = agent_pos + min(dist, 500) * dire
                pygame.draw.aaline(render_surface, "red" if i != 0 else "blue", array_to_vec(agent_pos), array_to_vec(cords))
        return distances
    
    def scan(self, agent_pos: np.ndarray, direction: np.ndarray):
        return min([ob.scan(agent_pos, direction) for ob in self.obstacles])

    def add_corner(self, corner:np.ndarray):
        if self.cur_ob == None:
            self.cur_ob = Obstacle()
        self.cur_ob.corners.append(corner)

    def finish_obstacle(self):
        self.cur_ob.finished = True
        self.obstacles.append(self.cur_ob)
        self.cur_ob = None
    
