import pygame
from environment import Obstacle, Environment
from enum import Enum, auto
import numpy as np


class MouseMode(Enum):
    Robot = auto()
    Object = auto()
    Goal = auto()

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
