import pygame
import numpy as np
from ui import MouseMode
from environment import Environment
from os import listdir
from enum import Enum, auto
import sys
from pandas import read_hdf
from run_util import *


# HDLS = any([s in sys.argv for s in ["-h", "-H", "headless", "Headless"]])
HDLS = False
print(HDLS)

# Run Options
data_file = None

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
    target_fps = 10
    dt = 1 / target_fps
else:
    dt = animation_data.iloc[1].name - animation_data.iloc[0].name
    target_fps = 1 / dt
v = 0
w = 0
