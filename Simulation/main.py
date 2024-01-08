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
    
    # Close and Save
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        break
    if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
        ENV.to_json()

    # MouseMode
    if keys[pygame.K_F1]:
        MODE = MouseMode.Robot
    elif keys[pygame.K_F2]:
        MODE = MouseMode.Object
    elif keys[pygame.K_F3]:
        MODE = MouseMode.Goal

    if CTRL == ControllMode.Player:
        v, w = player_controll(keys, v, w)
    elif CTRL == ControllMode.Controller:
        v,w = controll(ENV)
    elif CTRL == ControllMode.Animation:
        t = animation_controll(ENV)
        if t < 0:
            break

    
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
    pygame.display.flip()

ENV.finish_up()
pygame.quit()