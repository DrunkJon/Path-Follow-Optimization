from torch import NoneType
from environment import Environment
import numpy as np
from Turtlebot_Kinematics import translate_differential_drive, move_turtle
from time import time

def timer(func):
    def inner(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(func.__name__, time() - start)
        return result
    return inner

def translate_vector(vec):
    out = []
    for i in range(0,len(vec), 2):
        sub_vec = vec[i:i+2]
        trans_v = translate_differential_drive(*sub_vec)
        out.append(trans_v)
    return out

def disturb(vec):
    return vec * np.random.rand(len(vec))


####### Single PSO ######
class PSO_Controller:
    def __init__(self, samples=20, max_v=22.2, min_v=-22.2, dt=0.05) -> None:
        self.samples = samples
        self.max_v = max_v
        self.min_v = min_v
        self.dt = dt

    def __call__(self, env: Environment):
        pass
    
    def apply_swarm_forces(self, ind, ind_best, global_best, velocity, inertia=0.5, cog=0.5, soc=0.2):
        new_velocity = velocity * inertia + cog * disturb(ind_best - ind) + soc * disturb(global_best - ind) 
        new_ind = ind + new_velocity
        for i in range(len(new_ind)):
            val = new_ind[i]
            if val >= self.max_v:
                new_ind[i] = self.max_v
                new_velocity[i] = 0
            elif val <= self.min_v:
                new_ind[i] = self.min_v
                new_velocity[i] = 0
        return new_ind, new_velocity


####### Multi PSO ######
class Multi_PSO_Controller(PSO_Controller):
    def __init__(self, samples=20, max_v=22.2, min_v=-22.2, horizon=10, dt=0.05) -> None:
        super().__init__(samples, max_v, min_v, dt)
        self.horizon = horizon
        self.pop = self.gen_swarm()

    @timer
    def __call__(self, env: Environment, iterations=20, sensor_fusion=None):
        global_best, global_fit = self.particle_swarm_optimization(env, pop=self.pop, iterations=iterations,sensor_fusion=sensor_fusion)
        # print(f"fit={global_fit} | {global_best[:2]}")
        self.next_pop()
        return translate_differential_drive(*global_best[:2])
    
    def next_pop(self):
        for i in range(len(self.pop)):
            self.pop[i][:-2] = self.pop[i][2:]
            self.pop[i][-2:] = self.generate_v_vector(horizon=1)[:]
    
    def generate_v_vector(self, horizon=0):
        if horizon <= 0:
            horizon = self.horizon
        v_range = 2 * self.max_v
        vec = np.array([self.min_v]*horizon*2) + np.random.rand(horizon*2) * v_range
        return vec

    def fitness(self, vec, env:Environment, discount=0.95, sensor_fusion=None):
        trans_v_list = translate_vector(vec)
        positions = []
        cur_state = env.get_internal_state()
        if type(sensor_fusion) == NoneType:
            sensor_fusion = env.get_sensor_fusion()
        overall_fit = 0
        for i, (v, w) in enumerate(trans_v_list):
            cur_pos = move_turtle(cur_state, v, w, self.dt)
            positions.append(cur_pos)
            fit = env.fitness_single(cur_pos[0:2], sensor_fusion=sensor_fusion)
            if fit >= env.collision_penalty:
                overall_fit += env.collision_penalty * (len(trans_v_list) - i)
            else: overall_fit += fit*discount**i
        return overall_fit

    def gen_swarm(self):
        return [self.generate_v_vector(self.horizon) for _ in range(self.samples)]

    @timer
    def particle_swarm_optimization(self, env:Environment, pop, charged_pop=[], iterations=100, sensor_fusion=None):
        start = time()
        turbulence_trigger = 0.5
        # ATTENTION!: fitness is minimized
        if sensor_fusion == None:
            sensor_fusion = env.get_sensor_fusion()
        fit = lambda ind: self.fitness(ind, env, sensor_fusion=sensor_fusion)
        # TODO: make compatible with charged particles
        velocities = [np.zeros_like(ind) for ind in pop]
        individual_best = list(pop)
        individual_fit = [fit(ind) for ind in individual_best]
        global_best = individual_best[np.argmin(individual_fit)]
        global_fit = np.min(individual_fit)
        print("setup:", time() - start)
        for iteration in range(iterations):
            start1 = time()
            new_pop = []
            for i,ind in enumerate(pop):
                start2 = time()
                if turbulence_trigger >= np.linalg.norm(velocities[i]) and iteration > 0:
                    new_ind = self.generate_v_vector(self.horizon)
                else:
                    new_ind, velocities[i] = self.apply_swarm_forces(ind, individual_best[i], global_best, velocities[i])
                new_pop.append(new_ind)
                new_fit = fit(new_ind)
                if new_fit < individual_fit[i]:
                    individual_fit[i] = new_fit
                    individual_best[i] = new_ind
                    if new_fit < global_fit:
                        global_fit = new_fit
                        global_best = new_ind
                # print("inner_loop:", time() - start2)
            self.pop = new_pop
            print("main_loop:", time() - start1)
        return global_best, global_fit