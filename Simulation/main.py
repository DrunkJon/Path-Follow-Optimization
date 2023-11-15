import pygame
import numpy as np
from Turtlebot_Kinematics import *
from environment import *
from ui import MouseMode, mouse_action


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

display_center = np.array([screen.get_width() / 2, screen.get_height() / 2])
v = 0
w = 0
MODE = MouseMode.Robot
old_presses = (False, False, False)

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
ENV = Environment()
ENV.add_corner(np.array([150, 150]))
ENV.add_corner(np.array([575, 100]))
ENV.add_corner(np.array([1090, 300]))
ENV.add_corner(np.array([660, 350]))
ENV.add_corner(np.array([540, 150]))
ENV.add_corner(np.array([120, 200]))
ENV.finish_obstacle()

ENV.add_corner(np.array([100, 650]))
ENV.add_corner(np.array([350, 600]))
ENV.add_corner(np.array([220, 700]))

while True:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if not running: break

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    ENV.render(screen)
    ENV.get_distance_scans(render_surface=screen)
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        v = clamp(v + 5 * dt, -25, 25)
    if keys[pygame.K_s]:
        v = clamp(v - 5 * dt, -25, 25)
    if keys[pygame.K_a]:
        w = clamp(w + np.pi / 12 * dt, -np.pi / 8, np.pi / 8)
    if keys[pygame.K_d]:
        w = clamp(w - np.pi / 12 * dt, -np.pi / 8, np.pi / 8)
    if keys[pygame.K_F1]:
        MODE = MouseMode.Robot
    elif keys[pygame.K_F2]:
        MODE = MouseMode.Object
    elif keys[pygame.K_F3]:
        MODE = MouseMode.Goal

    presses = pygame.mouse.get_pressed(3)
    clicks = tuple(new and not old for new, old in zip(presses, old_presses))
    mouse_action(clicks, screen, MODE, ENV)
    old_presses = presses

    ENV.update_robo_state(v, w, dt)
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Mode:{mode_str()}', True, "black")
    screen.blit(img, (20, 20))

    # print(f"v: {v} w:{w}")

    # render_window(screen, robo_state, v, 5)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(30) / 1000

pygame.quit()