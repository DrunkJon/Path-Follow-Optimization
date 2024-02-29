import pygame
from environment import Obstacle, Environment
from enum import Enum, auto
import numpy as np
from Turtlebot_Kinematics import rotate


def array_to_vec(vec: np.ndarray) -> pygame.Vector2:
    return pygame.Vector2(vec[0], vec[1])

##### MOUSE CONTROLS #####

class MouseMode(Enum):
    Robot = auto()
    Object = auto()
    Goal = auto()

    def to_str(self):
        if self == MouseMode.Robot:
            return "Robot"
        elif self == MouseMode.Object:
            return "Object"
        elif self == MouseMode.Goal:
            return "Goal"

def mouse_action(clicks, presses, mode: MouseMode, env: Environment):
    left_clicked, middle_clicked, right_clicked = clicks
    left_pressed, middle_pressed, right_pressed = presses
    x, y = pygame.mouse.get_pos()
    if mode == MouseMode.Robot:
        if left_pressed:
            env.set_robo_state(np.array([x, y, env.get_robo_angle()], dtype=float))
        if right_pressed:
            env.turn_robo_towards(np.array([x, y], dtype=float))
    if mode == MouseMode.Object:
        if left_clicked:
            env.add_corner(np.array([x, y], dtype=float))
        if right_clicked:
            env.finish_obstacle()
    if mode == MouseMode.Goal:
        if left_pressed:
            env.set_goal_pos(np.array([x, y], dtype=float))

##### ENVIRONMENT RENDER #####

def render_obstacle(ob: Obstacle, surface: pygame.Surface):
    if len(ob.corners) >= 2:
        points = [array_to_vec(a) for a in ob.translate_corners()]
        if ob.finished:
            pygame.draw.polygon(surface, "grey", points)
        pygame.draw.aalines(surface, "black", ob.finished, points)
    else:
        for corner in ob.translate_corners():
            pygame.draw.circle(surface, "black", array_to_vec(corner), 5)

def render_robo(env: Environment, surface: pygame.Surface):
    robo_vec = array_to_vec(env.robo_state)
    pygame.draw.circle(surface, "black", robo_vec, 15)
    direction_delta = rotate(np.array([15,0]), env.robo_state[2])
    line_end = pygame.Vector2(robo_vec.x + direction_delta[0], robo_vec.y + direction_delta[1])
    pygame.draw.aaline(surface, "white", robo_vec, line_end)

def render_environment(env: Environment, surface: pygame.Surface):
    for ob in env.map_obstacles:
        render_obstacle(ob, surface)
    if env.cur_ob != None:
        env.cur_ob.render(surface)
    # render goal
    pygame.draw.circle(surface, "yellow", array_to_vec(env.goal_pos), 10)
    pygame.draw.circle(surface, "black", array_to_vec(env.goal_pos), 10, 2)    
    # render robot
    render_robo(env, surface)

def render_scanlines(distances, env: Environment, surface: pygame.Surface, skip = 3):
    directions = [
        rotate(np.array([1.0,0.0]), angle) 
        for angle in 
        np.linspace(env.get_robo_angle(), np.pi * 2 + env.get_robo_angle(), 360)
    ]
    for i, (dist, dire) in enumerate(zip(distances, directions)):
        if i % skip != 0: continue
        cords = env.get_robo_pos() + min(dist, 500) * dire
        pygame.draw.aaline(surface, "red" if i != 0 else "blue", array_to_vec(env.get_robo_pos()), array_to_vec(cords))
        render_robo(env, surface)