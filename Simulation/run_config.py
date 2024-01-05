import pygame
import numpy as np
from ui import MouseMode
from environment import Environment
from os import listdir

# env values
std_env = "simple"
record = False

# simulation values
target_fps = 20
dt = 1 / target_fps
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