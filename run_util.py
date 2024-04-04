from environment import Environment
from os import listdir
import numpy as np

def load_ENV(filename, record):
    env_files = sorted(listdir("levels"), reverse=True)
    if filename != None and f"{filename}.json" in env_files:
        print(f"loading {filename}")
        with open(f"./levels/{filename}.json", "r") as file:
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
    return ENV

def clamp(val, min= None, max= None):
    if min != None and val < min:
        return min
    if max != None and val > max:
        return max
    else:
        return val