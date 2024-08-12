import numpy as np
import json

from pygame import ver

ROBO_RADIUS = 16

def rotation_matrix(angle):
    return np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]], dtype=float)

def obstacle_from_list(points: list) -> dict:
    offset = points[0]
    point_arrays = [np.array(p, dtype=float) for p in points]
    offset_array = np.array(offset, dtype=float)
    point_arrays = [p - offset_array for p in point_arrays]
    corners = [list(p) for p in point_arrays]
    return {
        "offset": offset,
        "corners": corners
    }

def rectangle(start, width: float, height:float, angle = 0.0):
    start_vec = np.array(start, dtype=float)
    r = rotation_matrix(angle)
    w_vec = r @ np.array([width, 0.], dtype=float)
    h_vec = r @ np.array([0., height], dtype=float)
    point_vecs = [
        start_vec, 
        start_vec + w_vec, 
        start_vec + w_vec + h_vec, 
        start_vec + h_vec
    ]
    points = [list(p) for p in point_vecs]
    return obstacle_from_list(points)
    

def base_map():
    # map is 600 by 600
    # robo starts middle left, goal is middle right
    # map is surrounded by walls
    _map = {}
    _map["robo_state"] = [50, 300, 0]
    _map["goal_pos"] = [550, 300]
    _map["obstacles"] = [
        rectangle([-100, -100], 800, 100),
        rectangle([-100, -100], 100, 800),
        rectangle([700, 700], -800, -100),
        rectangle([700, 700], -100, -800),
    ]
    _map["unknowns"] = []
    return _map

def tight_map(tightness = 2.5, length = 100.0, vertical_offset = 0.0, unknown=True):
    _map = base_map()
    start_point1 = [300 - length / 2 , 0]
    start_point2 = [300 - length / 2 , 600]
    height = 300 - tightness * ROBO_RADIUS
    _map["unknowns" if unknown else "obstacles"].append(rectangle(start_point1, length, height - vertical_offset))
    _map["unknowns" if unknown else "obstacles"].append(rectangle(start_point2, length, -height - vertical_offset))
    return _map

def wall_map(height = 50.0, depth = 0.0, unknown=True):
    _map = base_map()
    _map["unknowns" if unknown else "obstacles"].append(obstacle_from_list([
        [300 - depth, 300 - height],
        [300, 300],
        [300 - depth, 300 + height],
        [330, 300]
    ]))
    return _map

def cluttered_map(objects= 5, size = 50, unknown=True):
    _map = base_map()
    for _ in range(objects):
        start_point = list(np.random.rand(2) * 400 + np.array([100, 100]))
        _map["unknowns" if unknown else "obstacles"].append(
            rectangle(start_point, size, size, np.random.rand() * 2*np.pi)
        )
    return _map

def save_map(_map, name:str, directory = "./levels"):
    try:
        with open(f"{directory}/{name}.json", "w") as file:
            json.dump(_map, file, indent=3)
    except Exception as err:
        raise Exception(f"{err}\n\t was caused by map:\n\t{_map}")


if __name__ == "__main__":
    # _map = tight_map(length=150, tightness=2.0, vertical_offset=20.0)
    # save_map(_map, "tight_map")
    # _map = wall_map(depth= 15)
    # save_map(_map, "wall_map")
    _map = cluttered_map(objects=8, size=50, unknown=False)
    save_map(_map, "cluttered_map")