import pygame
import numpy as np
from Turtlebot_Kinematics import *
from environment import *

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

display_center = np.array([screen.get_width() / 2, screen.get_height() / 2])
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
v = 0
w = 0
robo_state = np.array([display_center[0],display_center[1],0])

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
ENV.add_corner(np.array([50, 150]))
ENV.add_corner(np.array([175, 100]))
ENV.add_corner(np.array([290, 300]))
ENV.add_corner(np.array([160, 350]))
ENV.add_corner(np.array([140, 150]))
ENV.add_corner(np.array([20, 200]))
ENV.finish_obstacle()

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
    pygame.draw.circle(screen, "red", player_pos, 15)
    line_end = rotate(np.array([15,0]), robo_state[2])
    pygame.draw.aaline(screen, "white", player_pos, pygame.Vector2(player_pos.x + line_end[0], player_pos. y + line_end[1]))
    ENV.get_distance_scans(np.array(robo_state[:2]), robo_state[2], render_surface=screen)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        v = clamp(v + 5 * dt, -25, 25)
    if keys[pygame.K_s]:
        v = clamp(v - 5 * dt, -25, 25)
    if keys[pygame.K_a]:
        w = clamp(w + np.pi / 12 * dt, -np.pi / 8, np.pi / 8)
    if keys[pygame.K_d]:
        w = clamp(w - np.pi / 12 * dt, -np.pi / 8, np.pi / 8)

    robo_state = move_turtle(robo_state, v, w, dt)
    player_pos = array_to_vec(robo_state)
    print(f"v: {v} w:{w}")

    render_window(screen, robo_state, v, 5)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()