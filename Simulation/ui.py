import pygame
from environment import Obstacle, Environment
from enum import Enum, auto
import numpy as np


class MouseMode(Enum):
    Robot = auto()
    Object = auto()
    Goal = auto()

def mouse_action(clicks, surface: pygame.Surface, mode: MouseMode, env: Environment):
    left_clicked, middle_clicked, right_clicked = clicks
    if mode == MouseMode.Robot:
        if left_clicked:
            x, y = pygame.mouse.get_pos()
            env.set_robo_state(np.array([x, y, env.get_robo_angle()]))
        if right_clicked:
            # make robot face mouse
            # ! ignore if mouse pos == robo pos
            pass
    if mode == MouseMode.Object:
        if left_clicked:
            x, y = pygame.mouse.get_pos()
            env.add_corner(np.array([x, y]))
        if right_clicked:
            env.finish_obstacle()
