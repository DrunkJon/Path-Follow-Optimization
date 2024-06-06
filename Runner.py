from environment import Environment
from enum import Enum, auto
from controller import Controller
from dwa_controller import DWA_Controller

class ControllMode(Enum):
    Player = auto()
    Controller = auto()
    DWA = auto()
    MultiPSO = auto()
    PSO = auto()
    Animation = auto()


class Runner():
    def __init__(self, ENV:Environment, CTRL:ControllMode, controller=None, max_step=1000, dt=0.1) -> None:
        self.step_count = 0
        self.max_step = max_step
        self.ENV = ENV
        self.finished = False
        self.distances = []
        self.v, self.w = 0, 0
        self.dt = dt
        self.CTRL = CTRL
        self.controller = controller
        if CTRL == ControllMode.Controller:
            assert controller != None
        self.apply_control = True
    

    def step(self, verbose = True):
        self.step_count += 1
        if verbose: print("#",self.step_count)
        # gets combined sensor and map data for various simulations
        self.sensor = self.ENV.get_sensor_fusion()
        if self.sensor.is_empty: raise Exception("Empty sensor fusion")
        # online data collection of closest obstacles
        self.distances.append(self.ENV.get_obstacle_dist(sensor_fusion=self.sensor))

        
        if self.CTRL == ControllMode.DWA:
            self.v, self.w = self.controller(self.ENV, dt=2.0, sensor_fusion=self.sensor)  
        elif self.CTRL == ControllMode.MultiPSO:
            self.v, self.w = self.controller(self.ENV, sensor_fusion=self.sensor, true_dt=self.dt)
        elif self.CTRL == ControllMode.PSO:
            self.v, self.w = self.controller(self.ENV, sensor_fusion=self.sensor)
        
        # next simulation step
        if self.apply_control:
            self.ENV.step(self.v, self.w, self.dt)

        # determine if finished
        if self.step_count >= self.max_step or self.ENV.get_dist_to_goal() < self.ENV.robo_radius:
            print(f"finished at step #{self.step_count}\ndistance to goal: {self.ENV.get_dist_to_goal()}")
            self.finished = True
            
            
    def loop(self):
        while not self.finished:
            # simulation step
            self.step()
        # exports env data
        self.ENV.finish_up()
        
if __name__ == "__main__":
    from dwa_controller import DWA_Controller
    from pso_controller import Multi_PSO_Controller, PSO_Controller
    from environment import load_ENV
    from Turtlebot_Kinematics import difDriveKin, unicycleKin

    kinematic = difDriveKin()

    ENV = load_ENV("Corner Long", kinematic, record=False)

    # length of one simulation tick
    dt = 0.1
    # length of time step of Optimizers
    virtual_dt = 1.5
    # look ahead steps for MultiPSO | total lookahead time is virtual_dt * horizon
    horizon = 5
    # iterations for (Multi)PSO Controller
    iterations = 7
    ### type: DWA; MultiPSO; PSO
    CTRL = ControllMode.MultiPSO

    if CTRL == ControllMode.DWA:
        controller = DWA_Controller()
    elif CTRL == ControllMode.MultiPSO:
        controller = Multi_PSO_Controller(10, kinematic, horizon, virtual_dt, iterations)
    elif CTRL == ControllMode.PSO:
        controller = PSO_Controller(10, kinematic, dt)
        horizon = 1
    else:
        raise Exception(f"<{CTRL}> is not a valid ctrl type for headless")
    
    # Headless Runner
    runner = Runner(ENV, CTRL, controller, max_step=1000, dt=dt)
    runner.loop()