import pygame
import numpy as np
from Turtlebot_Kinematics import *
from environment import *
from ui import MouseMode, mouse_action
from os import listdir
from simple_controller import controll


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

display_center = np.array([screen.get_width() / 2, screen.get_height() / 2])
v = 0
w = 0
MODE = MouseMode.Robot
old_presses = (False, False, False)
target_fps = 20
dt = 1 / target_fps
font = pygame.font.SysFont(None, 24)

def mode_str():
    if MODE == MouseMode.Robot:
        return "Robot"
    elif MODE == MouseMode.Object:
        return "Object"
    elif MODE == MouseMode.Goal:
        return "Goal"
    else:
        raise ValueError("variable mode is not of enum type MouseMode")


def clamp(val, min= None, max= None):
    if min != None and val < min:
        return min
    if max != None and val > max:
        return max
    else:
        return val
    
def array_to_vec(vec: np.array) -> pygame.Vector2:
    return pygame.Vector2(vec[0], vec[1])
    
def render_window(surface: pygame.Surface, state: np.array, v = 25, dt = 1): 
    last_point = None
    first_point = None
    for v_ in (v, -v):
        for w in np.linspace(-np.pi, np.pi, 60):
            next_state = move_turtle(state, v_, w, dt)
            cur_point = array_to_vec(next_state)
            if last_point == None:
                first_point = cur_point
            else:
                pygame.draw.aaline(surface, "blue", last_point, cur_point)
            last_point = cur_point
    pygame.draw.aaline(surface, "blue", last_point, first_point)

# environment setup
env_files = sorted(listdir("levels"), reverse=True)
std_env = "simple"
record = False
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

while True:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if not running: break

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # ENV.get_distance_scans(render_surface=screen)
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        break
    if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
        ENV.to_json()
    if keys[pygame.K_w]:
        v = clamp(v + 15 * dt, -75, 75)
    if (not keys[pygame.K_LCTRL]) and keys[pygame.K_s]:
        v = clamp(v - 15 * dt, -75, 75)
    if keys[pygame.K_a]:
        w = clamp(w + np.pi / 5 * dt, -np.pi / 3, np.pi / 3)
    if keys[pygame.K_d]:
        w = clamp(w - np.pi / 5 * dt, -np.pi / 3, np.pi / 3)
    if keys[pygame.K_F1]:
        MODE = MouseMode.Robot
    elif keys[pygame.K_F2]:
        MODE = MouseMode.Object
    elif keys[pygame.K_F3]:
        MODE = MouseMode.Goal

    v,w = controll(ENV)

    presses = pygame.mouse.get_pressed(3)
    clicks = tuple(new and not old for new, old in zip(presses, old_presses))
    mouse_action(clicks, presses, MODE, ENV)
    old_presses = presses

    ENV.step(v, w, dt, screen)

    img = font.render(f'Mode:{mode_str()}', True, "black")
    screen.blit(img, (20, 20))

    fps = round(1 / (clock.tick(target_fps) / 1000), 1)
    img = font.render(f'fps:{fps} | target:{target_fps}', True, "black")
    screen.blit(img, (1260 - img.get_width(), 20))
    # print(f"v: {v} w:{w}")

    # render_window(screen, robo_state, v, 5)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    
ENV.finish_up()
pygame.quit()