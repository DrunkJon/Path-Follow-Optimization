from Runner import Runner, Environment, ControllMode
import pygame
from Turtlebot_Kinematics import droneKin
from ui import *


# util functions
def clamp(val, min= None, max= None):
    if min != None and val < min:
        return min
    if max != None and val > max:
        return max
    else:
        return val


def player_controll(keys, v, w, dt):
    if keys[pygame.K_w]:
        v = clamp(v + 15 * dt, -75, 75)
    if (not keys[pygame.K_LCTRL]) and keys[pygame.K_s]:
        v = clamp(v - 15 * dt, -75, 75)
    if keys[pygame.K_a]:
        w = clamp(w + np.pi / 2 * dt, -np.pi / 3, np.pi / 3)
    if keys[pygame.K_d]:
        w = clamp(w - np.pi / 2 * dt, -np.pi / 3, np.pi / 3)
    return v, w


def animation_controll(ENV, animation_gen):
    try:
        t, robo_state, goal_pos = next(animation_gen)
        ENV.set_robo_state(robo_state)
        ENV.set_goal_pos(goal_pos)
        return t
    except StopIteration:
        return -1


class UI_Runner(Runner):
    def __init__(self, ENV: Environment, CTRL: ControllMode, controller=None, max_step=1000, dt=0.1, apply_control=True) -> None:
        super().__init__(ENV, CTRL, controller, max_step, dt)
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 900))
        self.fit_surface = self.screen.copy()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.apply_control = apply_control
    
    def loop(self, visualize_fitness=True):
        if visualize_fitness and CTRL == ControllMode.MultiPSO:
            render_fitness(self.ENV, self.controller, self.fit_surface)
            
        old_presses = (False, False, False)
        MODE = MouseMode.Robot
    
        while True:
            flag = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT: flag = True
            if flag: break 
                
            if visualize_fitness and CTRL == ControllMode.MultiPSO:
                self.screen.blit(self.fit_surface, (0,0))
            else:
                self.screen.fill("white")
                
            # Close and Save
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
            if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
                self.ENV.to_json()
            # change whether control should be applied
            if keys[pygame.K_PERIOD]:
                self.apply_control = True
            elif keys[pygame.K_COMMA]:
                self.apply_control = False

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
                self.v, self.w = player_controll(keys, self.v, self.w, self.dt)
            elif self.CTRL == ControllMode.Animation:
                pass
            
            presses = pygame.mouse.get_pressed(3)
            clicks = tuple(new and not old for new, old in zip(presses, old_presses))
            mouse_action(clicks, presses, MODE, self.ENV)
            old_presses = presses
            
            self.step()
            if type(self.controller.kinematic) == unicycleAcceleration:
                print("velocities:", self.controller.kinematic.v1, self.controller.kinematic.v2)
            
            # render potential movement radius
            if self.CTRL in [ControllMode.MultiPSO, ControllMode.PSO]:
                render_radius(self.ENV.get_internal_state(), 22.2 * self.controller.horizon * self.controller.dt, self.screen)
            # render internal robo position in blue
            render_robo(self.ENV.get_internal_state(), self.ENV.robo_radius, self.screen, color="blue")
            # renders obstacles, goal and actual robo position
            render_environment(self.ENV, self.screen)
            # creates red overlay of where robot thinks obstacles are
            render_sensor_fusion(self.ENV, self.screen, sensor_fusion=self.sensor)
            # blit(left_sub_screen, temp_surface, ENV.get_robo_pos())
            if self.CTRL in [ControllMode.MultiPSO, ControllMode.PSO]:
                render_particle_trajectories(self.ENV, self.controller, self.screen)

            if self.CTRL == ControllMode.DWA:
                if visualize_fitness:
                    live_dwa_fitness_redner(self.ENV, self.controller, self.screen)
                render_dwa_trajectory(self.ENV, self.controller, self.screen, self.v, self.w)

            render_dir_line(self.screen, self.ENV.get_internal_state(), self.ENV.robo_radius)

            img = self.font.render(f'Mode:{MODE.to_str()}', True, "black")
            self.screen.blit(img, (20, 20))

            fps = round(1 / (self.clock.tick(1 / self.dt) / 1000), 1)
            img = self.font.render(f'fps:{fps} | target:{1 / self.dt}', True, "black")
            self.screen.blit(img, (1260 - img.get_width(), 20))

            pygame.display.flip()
        # export ENV data if wanted
        self.ENV.finish_up()
        # close pygame window
        pygame.quit()


if __name__ == "__main__":
    from dwa_controller import DWA_Controller
    from pso_controller import Multi_PSO_Controller, PSO_Controller
    from environment import load_ENV
    from Turtlebot_Kinematics import difDriveKin, unicycleKin, AnimationModel, unicycleAcceleration
    import json

    kinematic = unicycleKin()
    # kinematic.v1_min = -5.0

    map_path = "data\Map Experiment #2\9\cluttered_8obs_50x50_known.json"
    ENV = Environment.from_json_file(map_path, kinematic, record=False)
    ENV.use_errors = True

    # length of one simulation tick
    dt = 0.1
    # length of time step of Optimizers
    virtual_dt = 0.75
    # look ahead steps for MultiPSO | total lookahead time is virtual_dt * horizon
    horizon = 7
    ### type: DWA; MultiPSO; PSO; Player
    CTRL = ControllMode.MultiPSO

    if CTRL == ControllMode.DWA:
        controller = DWA_Controller(kinematic=kinematic, virtual_dt=2.0)
    elif CTRL == ControllMode.MultiPSO:
        controller = Multi_PSO_Controller(7, kinematic=kinematic, horizon=horizon, dt=virtual_dt, iterations=7)
    elif CTRL == ControllMode.PSO:
        controller = PSO_Controller(10, kinematic=kinematic, dt=virtual_dt)
        horizon = 1
    elif CTRL == ControllMode.Player or CTRL == ControllMode.Animation:
        controller = None
    else:
        raise Exception(f"<{CTRL}> is not a valid ctrl type for headless")

    # UI with Player Control
    runner = UI_Runner(ENV, CTRL, controller, dt=dt, apply_control=True)
    runner.loop(visualize_fitness=False)