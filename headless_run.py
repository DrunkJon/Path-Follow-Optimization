import pygame
import numpy as np
from ui import *
from dwa_controller import DWA_Controller
from pso_controller import Multi_PSO_Controller, PSO_Controller
from time import time


# contains all important variables and environment setup
from run_config import *

def main(ctrl_type, virtual_dt = 1.5, horizon=5, ENV=ENV):
    ### type: DWA; MultiPSO; SinglePSO
    if CTRL == ControllMode.Controller:
        if ctrl_type == "DWA":
            controller = Multi_PSO_Controller()
        elif ctrl_type == "MultiPSO":
            controller = Multi_PSO_Controller(10, 22.2, -22.2, horizon, virtual_dt)
        elif ctrl_type == "PSO":
            controller = PSO_Controller(10, 22.2, -22.2, dt, horizon)
            horizon = 1
        else:
            raise Exception(f"not a valid ctrl type ({ctrl_type})")

    if not HDLS and visualize_fitness:
        render_fitness(ENV, fit_surface)

    step = 0
    max_step = 1000
    distances = []
    running = True
    v, w = 0, 0
    while True:
        step += 1
        print("#",step)
        sensor = ENV.get_sensor_fusion()
        distances.append(ENV.get_obstacle_dist(sensor_fusion=sensor))
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        if not HDLS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running: break
        else:
            if step > max_step or ENV.get_dist_to_goal() < ENV.robo_radius:
                print(f"finished at step #{max_step}\ndistance to goal: {ENV.get_dist_to_goal()}")
                break

        # fill the screen with a color to wipe away anything from last frame
        if not HDLS:
            if visualize_fitness:
                screen.blit(fit_surface, (0,0))
            else:
                screen.fill("white")

        
        # Close and Save
        if not HDLS:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
            if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
                ENV.to_json()

        # MouseMode
        if not HDLS:
            if keys[pygame.K_F1]:
                MODE = MouseMode.Robot
            elif keys[pygame.K_F2]:
                MODE = MouseMode.Object
            elif keys[pygame.K_F3]:
                MODE = MouseMode.Unknown
            elif keys[pygame.K_F4]:
                MODE = MouseMode.Goal

        if CTRL == ControllMode.Player and not HDLS:
            v, w = player_controll(keys, v, w)
        elif CTRL == ControllMode.Controller or HDLS:
            if ctrl_type == "DWA":
                v, w = controller(ENV, 2.0)  
            elif ctrl_type == "MultiPSO":
                v,w = controller(ENV, iterations = 7, sensor_fusion=sensor, true_dt=dt)
            elif ctrl_type == "PSO":
                v,w = controller(ENV, iterations = 10, sensor_fusion=sensor)
        elif CTRL == ControllMode.Animation:
            t = animation_controll(ENV)
            if t < 0:
                break
        
        if not HDLS:
            presses = pygame.mouse.get_pressed(3)
            clicks = tuple(new and not old for new, old in zip(presses, old_presses))
            mouse_action(clicks, presses, MODE, ENV)
            old_presses = presses

        # distances = ENV.get_distance_scans()
        # render_scanlines(distances, ENV, parent_screen) 
        
        # next simulation step
        ENV.step(v, w, dt)
        # render potential movement radius
        if not HDLS:
            if ctrl_type in ["MultiPSO", "PSO"]:
                render_radius(ENV.get_internal_state(), 22.2 * horizon * virtual_dt, screen)
            # render intenal robo position in blue
            render_robo(ENV.get_internal_state(), ENV.robo_radius, screen, color="blue")
            # renders obstacles, goal and actual robo position
            render_environment(ENV, screen)
            # creates red overlay of where robot thinks obstacles are
            render_sensor_fusion(ENV, screen, sensor_fusion=sensor)
            # blit(left_sub_screen, temp_surface, ENV.get_robo_pos())
            if ctrl_type in ["MultiPSO", "PSO"]:
                render_particle_trajectories(ENV, controller, screen)

            img = font.render(f'Mode:{MODE.to_str()}', True, "black")
            screen.blit(img, (20, 20))

            fps = round(1 / (clock.tick(target_fps) / 1000), 1)
            img = font.render(f'fps:{fps} | target:{target_fps}', True, "black")
            screen.blit(img, (1260 - img.get_width(), 20))

            pygame.display.flip()

    ENV.finish_up()
    if not HDLS:
        pygame.quit()
    return step, distances

if __name__ == '__main__':
    main()