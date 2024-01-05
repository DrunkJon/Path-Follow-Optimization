import pygame
import numpy as np
from simple_controller import controll
from ui import mouse_action

pygame.init()

# contains all important variables and environment setup
from run_config import *


while True:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if not running: break

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    ENV.get_distance_scans(render_surface=screen)
    
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

    img = font.render(f'Mode:{MODE.to_str()}', True, "black")
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