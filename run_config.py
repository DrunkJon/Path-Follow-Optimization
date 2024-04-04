import pygame
import numpy as np
from ui import MouseMode
from environment import Environment
from os import listdir
from enum import Enum, auto
import sys
from pandas import read_hdf
from run_util import *

# env values
std_env = "simple copy"
record = False
ENV = load_ENV(std_env, record)

# Run Options
class ControllMode(Enum):
    Player = auto()
    Controller = auto()
    Animation = auto()

CTRL = ControllMode.Controller
data_file = None
HDLS = any([s in sys.argv for s in ["-h", "-H", "headless", "Headless"]])

animation_data = None
if CTRL == ControllMode.Animation:
    if data_file:
        animation_data = read_hdf(f"./data/{data_file}")
    else:
        animation_data = read_hdf(f"./data/{sorted(listdir('data'), reverse=True)[0]}")
    animation_gen = (
        (
            row.name, 
            np.array([row['robo_x'], row['robo_y'], row['robo_deg']]), 
            np.array([row['goal_x'], row['goal_y']])) 
        for row in animation_data.iloc
        )

# simulation values
if not CTRL == ControllMode.Animation:
    target_fps = 3
    dt = 1 / target_fps
else:
    dt = animation_data.iloc[1].name - animation_data.iloc[0].name
    target_fps = 1 / dt
v = 0
w = 0

# ui setup
parent_screen = pygame.display.set_mode((1600, 900))
left_sub_screen = parent_screen.subsurface(0, 0, parent_screen.get_width() / 2 - 1, parent_screen.get_height())
right_sub_screen = parent_screen.subsurface(parent_screen.get_width() / 2 + 1, 0, parent_screen.get_width() / 2 - 1, parent_screen.get_height())
visualize_fitness = False
fit_surface = parent_screen.copy()
map_size = (left_sub_screen.get_width(), left_sub_screen.get_height())
temp_surface = pygame.Surface(map_size)
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont(None, 24)
MODE = MouseMode.Robot
old_presses = (False, False, False)

# util functions
def player_controll(keys, v, w):
    if keys[pygame.K_w]:
        v = clamp(v + 15 * dt, -75, 75)
    if (not keys[pygame.K_LCTRL]) and keys[pygame.K_s]:
        v = clamp(v - 15 * dt, -75, 75)
    if keys[pygame.K_a]:
        w = clamp(w + np.pi / 2 * dt, -np.pi / 3, np.pi / 3)
    if keys[pygame.K_d]:
        w = clamp(w - np.pi / 2 * dt, -np.pi / 3, np.pi / 3)
    return v, w


def animation_controll(ENV):
    try:
        t, robo_state, goal_pos = next(animation_gen)
        ENV.set_robo_state(robo_state)
        ENV.set_goal_pos(goal_pos)
        return t
    except StopIteration:
        return -1