import pygame
import numpy as np
from ui import MouseMode
from environment import Environment
from os import listdir
from enum import Enum, auto
import sys
from pandas import read_hdf

# env values
std_env = "simple"
record = False

# Run Options
class ControllMode(Enum):
    Player = auto()
    Controller = auto()
    Animation = auto()

CTRL = ControllMode.Animation
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
    target_fps = 20
    dt = 1 / target_fps
else:
    dt = animation_data.iloc[1].name - animation_data.iloc[0].name
    target_fps = 1 / dt
v = 0
w = 0

# ui setup
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont(None, 24)
MODE = MouseMode.Robot
old_presses = (False, False, False)

# env load
env_files = sorted(listdir("levels"), reverse=True)
if std_env != None and f"{std_env}.json" in env_files:
    print(f"loading {std_env}")
    with open(f"./levels/{std_env}.json", "r") as file:
        json_str = file.read()
        ENV = Environment.from_json(json_str, record)
elif len(env_files) >= 1:
    print(f"loading {env_files[0]}")
    with open(f"./levels/{env_files[0]}", "r") as file:
        json_str = file.read()
        ENV = Environment.from_json(json_str, record)
else:
    print("loading new ENV")
    ENV = Environment(np.array([640,360,0], dtype=float), np.array([1260, 700], dtype=float), record)


# util functions
def clamp(val, min= None, max= None):
    if min != None and val < min:
        return min
    if max != None and val > max:
        return max
    else:
        return val


def player_controll(keys):
    if keys[pygame.K_w]:
        v = clamp(v + 15 * dt, -75, 75)
    if (not keys[pygame.K_LCTRL]) and keys[pygame.K_s]:
        v = clamp(v - 15 * dt, -75, 75)
    if keys[pygame.K_a]:
        w = clamp(w + np.pi / 5 * dt, -np.pi / 3, np.pi / 3)
    if keys[pygame.K_d]:
        w = clamp(w - np.pi / 5 * dt, -np.pi / 3, np.pi / 3)


def animation_controll(ENV):
    try:
        t, robo_state, goal_pos = next(animation_gen)
        ENV.set_robo_state(robo_state)
        ENV.set_goal_pos(goal_pos)
        return t
    except StopIteration:
        return -1