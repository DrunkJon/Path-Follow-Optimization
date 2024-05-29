import pygame
import numpy as np
from ui import *
from dwa_controller import DWA_Controller
from pso_controller import Multi_PSO_Controller, PSO_Controller
from time import time


# contains all important variables and environment setup
from run_config import *


virtual_dt = 1.5
horizon = 5
### type: DWA; MultiPSO; SinglePSO
ctrl_type = "DWA"
if CTRL == ControllMode.Controller:
    if ctrl_type == "DWA":
        controller = DWA_Controller()
    elif ctrl_type == "MultiPSO":
        controller = Multi_PSO_Controller(10, 22.2, -22.2, horizon, virtual_dt)
    elif ctrl_type == "PSO":
        controller = PSO_Controller(10, 22.2, -22.2, dt, horizon)
        horizon = 1
    else:
        raise Exception(f"not a valid ctrl type ({ctrl_type})")

class main():
    def __init__(self, ENV:Environment, CTRL:ControllMode, ctrl_type=None, controller=None, max_step=1000, dt=0.1) -> None:
        self.step_count = 0
        self.max_step = max_step
        self.ENV = ENV
        self.finished = False
        self.distances = []
        self.v, self.w = 0, 0
        self.dt = dt
        self.CTRL = CTRL
        self.controller = controller
        self.ctrl_type = ctrl_type
        if CTRL == ControllMode.Controller:
            assert controller != None
            assert ctrl_type != None
    
    def step(self, verbose = True):
        self.step_count += 1
        if verbose: print("#",self.step_count)
        # gets combined sensor and map data for various simulations
        self.sensor = ENV.get_sensor_fusion()
        # online data collection of closest obstacles
        self.distances.append(ENV.get_obstacle_dist(sensor_fusion=self.sensor))
        # determine if finished
        if self.step_count > self.max_step or self.ENV.get_dist_to_goal() < self.ENV.robo_radius:
            print(f"finished at step #{self.max_step}\ndistance to goal: {self.ENV.get_dist_to_goal()}")
            self.finished = True

        if self.CTRL == ControllMode.Controller:
            if self.ctrl_type == "DWA":
                self.v, self.w = controller(self.ENV, 2.0)  
            elif self.ctrl_type == "MultiPSO":
                self.v, self.w = controller(self.ENV, iterations = 7, sensor_fusion=self.sensor, true_dt=dt)
            elif self.ctrl_type == "PSO":
                self.v, self.w = controller(self.ENV, iterations = 10, sensor_fusion=self.sensor)
        
        # next simulation step
        self.ENV.step(self.v, self.w, self.dt)
            
            
    def loop(self):
        while not self.finished:
            # simulation step
            self.step()
        # exports env data
        ENV.finish_up()
 
class UI_main(main):
    def __init__(self, ENV: Environment, CTRL: ControllMode, ctrl_type=None, controller=None, max_step=1000, dt=0.1) -> None:
        super().__init__(ENV, CTRL, ctrl_type, controller, max_step, dt)
    
    def loop(self, screen: pygame.Surface, fit_surface: pygame.Surface, visualize_fitness=True):
        if visualize_fitness:
            render_fitness(ENV, fit_surface)
            
        old_presses = (False, False, False)
        MODE = MouseMode.Robot
    
        while True:
            flag = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT: flag = True
            if flag: break 
                
            if visualize_fitness:
                screen.blit(fit_surface, (0,0))
            else:
                screen.fill("white")
                
            # Close and Save
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
            if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
                self.ENV.to_json()

            # MouseMode
            if keys[pygame.K_F1]:
                MODE = MouseMode.Robot
            elif keys[pygame.K_F2]:
                MODE = MouseMode.Object
            elif keys[pygame.K_F3]:
                MODE = MouseMode.Unknown
            elif keys[pygame.K_F4]:
                MODE = MouseMode.Goal    
            
            if self.CTRL == ControllMode.Player:
                self.v, self.w = player_controll(keys, self.v, self.w)
            elif self.CTRL == ControllMode.Animation:
                t = animation_controll(ENV)
                if t < 0:
                    break
            
            presses = pygame.mouse.get_pressed(3)
            clicks = tuple(new and not old for new, old in zip(presses, old_presses))
            mouse_action(clicks, presses, MODE, self.ENV)
            old_presses = presses
            
            print(self.v, self.w)
            self.step()
            print(self.v, self.w)
            
            # render potential movement radius
            if self.ctrl_type in ["MultiPSO", "PSO"]:
                render_radius(self.ENV.get_internal_state(), 22.2 * self.controller.horizon * self.controller.dt, screen)
            # render intenal robo position in blue
            render_robo(self.ENV.get_internal_state(), self.ENV.robo_radius, screen, color="blue")
            # renders obstacles, goal and actual robo position
            render_environment(self.ENV, screen)
            # creates red overlay of where robot thinks obstacles are
            render_sensor_fusion(self.ENV, screen, sensor_fusion=self.sensor)
            # blit(left_sub_screen, temp_surface, ENV.get_robo_pos())
            if ctrl_type in ["MultiPSO", "PSO"]:
                render_particle_trajectories(self.ENV, self.controller, screen)

            img = font.render(f'Mode:{MODE.to_str()}', True, "black")
            screen.blit(img, (20, 20))

            fps = round(1 / (clock.tick(target_fps) / 1000), 1)
            img = font.render(f'fps:{fps} | target:{target_fps}', True, "black")
            screen.blit(img, (1260 - img.get_width(), 20))

            pygame.display.flip()
        # export ENV data if wanted
        ENV.finish_up()
        # close pygame window
        pygame.quit()
        
if __name__ == "__main__":
    # Headless
    # runner = main(ENV, CTRL, ctrl_type, controller)
    # UI with Player Control
    runner = UI_main(ENV, ControllMode.Player)
    runner.loop(screen, fit_surface, False)