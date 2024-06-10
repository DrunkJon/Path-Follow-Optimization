import pygame
from environment import Obstacle, Environment
from enum import Enum, auto
import numpy as np
from Turtlebot_Kinematics import rotate
import shapely
from pso_controller import Multi_PSO_Controller



def array_to_vec(vec: np.ndarray) -> pygame.Vector2:
    return pygame.Vector2(vec[0], vec[1])

##### MOUSE CONTROLS #####

class MouseMode(Enum):
    Robot = auto()
    Object = auto()
    Unknown = auto()
    Goal = auto()

    def to_str(self):
        if self == MouseMode.Robot:
            return "Robot"
        elif self == MouseMode.Object:
            return "Object"
        elif self == MouseMode.Unknown:
            return "Unknown"
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
            env.finish_obstacle(add_to_map=True)
    if mode == MouseMode.Unknown:
        if left_clicked:
            env.add_corner(np.array([x, y], dtype=float))
        if right_clicked:
            env.finish_obstacle(add_to_map=False)
    if mode == MouseMode.Goal:
        if left_pressed:
            env.set_goal_pos(np.array([x, y], dtype=float))

##### ENVIRONMENT RENDER #####

def render_obstacle(ob: Obstacle, surface: pygame.Surface, color = "grey"):
    if len(ob.corners) >= 2:
        points = [array_to_vec(a) for a in ob.translate_corners()]
        if ob.finished:
            pygame.draw.polygon(surface, color, points)
        pygame.draw.aalines(surface, "black", ob.finished, points)
    else:
        for corner in ob.translate_corners():
            pygame.draw.circle(surface, "black", array_to_vec(corner), 5)

def render_robo(robo_state: np.ndarray, radius: float, surface: pygame.Surface, color="black"):
    robo_vec = array_to_vec(robo_state)
    pygame.draw.circle(surface, color, robo_vec, radius)

def render_dir_line(surface: pygame.Surface, robo_state: np.array, radius:float, color="red"):
    robo_vec = array_to_vec(robo_state)
    direction_delta = rotate(np.array([radius,0]), robo_state[2])
    line_end = pygame.Vector2(robo_vec.x + direction_delta[0], robo_vec.y + direction_delta[1])
    pygame.draw.aaline(surface, color, robo_vec, line_end)

def render_radius(robo_state: np.ndarray, radius: float, surface: pygame.Surface, color="black"):
    robo_vec = array_to_vec(robo_state)
    pygame.draw.circle(surface, color, robo_vec, radius, 3)

def render_environment(env: Environment, surface: pygame.Surface, internal = False):
    for ob in env.map_obstacles:
        render_obstacle(ob, surface, color="grey")
    for ob in env.unknown_obstacles:
        render_obstacle(ob, surface, color="#c1ad53")
    if env.cur_ob != None:
        render_obstacle(env.cur_ob, surface, color="black")
    # render goal
    print("goal_pos", env.get_goal_pos())
    pygame.draw.circle(surface, "yellow", array_to_vec(env.get_goal_pos()), 10)
    pygame.draw.circle(surface, "black", array_to_vec(env.get_goal_pos()), 10, 2)    
    # render robot
    robo_state = env.robo_state if not internal else env.get_internal_state()
    render_robo(robo_state, env.robo_radius, surface)

def render_scanlines(distances, env: Environment, surface: pygame.Surface, skip = 12):
    directions = [
        rotate(np.array([1.0,0.0]), angle) 
        for angle in 
        np.linspace(env.get_robo_angle(), np.pi * 2 + env.get_robo_angle(), 360)
    ]
    for i, (dist, dire) in enumerate(zip(distances, directions)):
        if i % skip != 0: continue
        cords = env.get_robo_pos() + min(dist, env.max_scan_dist) * dire
        pygame.draw.aaline(surface, "red" if i != 0 else "blue", array_to_vec(env.get_robo_pos()), array_to_vec(cords))
        # render_robo(env, surface)

def blit(sub_screen:pygame.Surface, temp_surface: pygame.Surface, pos: np.ndarray):
    sub_screen.fill("gray")
    offset = tuple(np.array([sub_screen.get_width(), sub_screen.get_height()]) / 2 - pos)
    sub_screen.blit(temp_surface, (0, 0))

def retransform_sensor(poly: shapely.MultiPolygon, env: Environment):
    int_x, int_y, rho = env.internal_offset
    robo_x, robo_y = env.get_robo_pos()
    poly = shapely.affinity.rotate(poly, rho, origin=(robo_x + int_x, robo_y + int_y), use_radians=True)
    poly = shapely.affinity.translate(poly, xoff = -int_x, yoff = -int_y)
    return poly

def render_sensor_fusion(env: Environment, surface: pygame.Surface, sensor_fusion=None):
    # sensor fusion centered around internal state
    if sensor_fusion == None:
        sensor_fusion = env.get_sensor_fusion(point_cloud=True)
    # sensor fusion re-centered around actual state for visual clarity
    sensor_fusion = retransform_sensor(sensor_fusion, env)
    geoms = [sensor_fusion._geom] if not type(sensor_fusion) == shapely.GeometryCollection else sensor_fusion.geoms
    for geom in geoms:
        if type(geom) == shapely.Polygon:
            coords = list(geom.exterior.coords)
            pygame.draw.polygon(surface, "red", coords)
        elif type(geom) == shapely.Point:
            coords = tuple(geom.coords)
            pygame.draw.circle(surface, "red", coords, 3)

def interpolate(left, right, ratio):
    return ((1-ratio) * left + ratio * right)

def render_fitness(env: Environment, surface: pygame.Surface):
    vals = []
    max_val, min_val = -np.inf, np.inf
    sens = env.get_sensor_fusion()
    print("starting fitness calc...")
    spacing = 10
    w_offset, h_offset, _ = env.internal_offset
    for i_w in range(0, surface.get_width()+1, spacing):
        row = []
        for i_h in range(0, surface.get_height()+1, spacing):
            val = np.log(env.fitness_single(state=np.array((i_w+w_offset, i_h+h_offset, 0)), sensor_fusion=sens))
            if val > max_val: max_val = val
            elif val < min_val: min_val = val
            row.append(val)
            print(i_w, i_h, "->", val)
        vals.append(row)
    print(min_val, max_val)
    for i_w in range(surface.get_width()):
        for i_h in range(surface.get_height()):
            # val = vals[i_w][i_h]
            w_ratio = (i_w % spacing) / spacing
            h_ratio = (i_h % spacing) / spacing
            top_left = vals[i_w // spacing][i_h // spacing]
            top_right = vals[i_w // spacing + 1][i_h // spacing] if i_w // spacing + 1 <= len(vals) else top_left
            bot_left = vals[i_w // spacing][i_h // spacing + 1] if i_h // spacing + 1 < len(vals[0]) else top_left
            bot_right = vals[i_w // spacing + 1][i_h // spacing + 1] if i_h // spacing + 1 < len(vals[0]) else top_right
            val = interpolate(interpolate(top_left, top_right, w_ratio), interpolate(bot_left, bot_right, w_ratio), h_ratio)
            ratio = (val - min_val) / (max_val - min_val)
            # print("ratio", ratio)
            color = pygame.Color(int(ratio*255), int((1-ratio)*255), 200)
            surface.set_at((i_w, i_h), color)
    print("finished fitness calc")

def render_particle_trajectories(env:Environment, ctrl: Multi_PSO_Controller, surface: pygame.Surface):
    cur_state = env.get_internal_state()
    pop_offset = len(ctrl.pop)
    trajecories = [ctrl.get_trajectory(ind, cur_state) for ind in ctrl.individual_best[:pop_offset]]
    charged_trajecories = [ctrl.get_trajectory(ind, cur_state) for ind in ctrl.individual_best[pop_offset:]]
    for traj in charged_trajecories:
        traj = [state[:2] for state in traj]
        pygame.draw.aalines(surface, "grey", False, traj)
    for traj in trajecories:
        traj = [state[:2] for state in traj]
        pygame.draw.aalines(surface, "orange", False, traj)
    best_traj = [state[:2] for state in ctrl.get_trajectory(ctrl.global_best, cur_state)]
    pygame.draw.aalines(surface, "green", False, best_traj)