import numpy as np
from typing import List
import pygame
from Turtlebot_Kinematics import rotate, move_turtle


def array_to_vec(vec: np.ndarray) -> pygame.Vector2:
    return pygame.Vector2(vec[0], vec[1])

def vec_angle(v: np.ndarray, u: np.ndarray) -> float:
    return np.arccos((v @ u) / (np.linalg.norm(v) * np.linalg.norm(u)))


class Obstacle:
    corners: List[np.ndarray]

    def __init__(self) -> None:
        self.corners = []
        self.finished = False

    def render(self, surface: pygame.Surface):
        if len(self.corners) >= 2:
            points = [array_to_vec(a) for a in self.corners]
            if self.finished:
                pygame.draw.polygon(surface, "grey", points)
            pygame.draw.aalines(surface, "black", self.finished, points)
        else:
            for corner in self.corners:
                pygame.draw.circle(surface, "black", array_to_vec(corner), 5)

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
    robo_state: np.ndarray

    def __init__(self) -> None:
        self.obstacles = []
        self.cur_ob = None
        self.robo_state = np.array([640,360,0])

    def render(self, surface: pygame.Surface):
        for ob in self.obstacles:
            ob.render(surface)
        if self.cur_ob != None:
            self.cur_ob.render(surface)
        self.render_robo(surface)

    def render_robo(self, surface: pygame.Surface):
        robo_vec = array_to_vec(self.robo_state)
        pygame.draw.circle(surface, "black", robo_vec, 15)
        direction_delta = rotate(np.array([15,0]), self.robo_state[2])
        line_end = pygame.Vector2(robo_vec.x + direction_delta[0], robo_vec.y + direction_delta[1])
        pygame.draw.aaline(surface, "white", robo_vec, line_end)

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
            self.cur_ob = Obstacle()
        self.cur_ob.corners.append(corner)

    def finish_obstacle(self):
        self.cur_ob.finished = True
        self.obstacles.append(self.cur_ob)
        self.cur_ob = None
    
