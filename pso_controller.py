from types import NoneType
from environment import Environment
import numpy as np
from Turtlebot_Kinematics import translate_differential_drive, move_turtle
from time import time
from functions import function_3, sigmoid
import shapely

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


####### Multi PSO ######
class Multi_PSO_Controller():
    v_dim = 2
    def __init__(self, samples=10, max_v=22.2, min_v=-22.2, horizon=10, dt=0.05) -> None:
        self.samples = samples
        self.max_v = max_v
        self.min_v = min_v
        self.dt = dt
        self.horizon = horizon
        self.pop = self.gen_swarm()
        self.charged_pop = self.gen_swarm()
        self.velocities = [np.zeros_like(ind) for ind in self.pop]
        self.individual_best = list(self.pop)
        self.individual_fit = [np.infty for _ in self.individual_best]
        self.global_best = self.individual_best[np.argmin(self.individual_fit)]
        self.global_fit = np.min(self.individual_fit)

    @timer
    def __call__(self, env: Environment, iterations=20, true_dt=0.05, sensor_fusion=None, verbose=True):
        if type(sensor_fusion) == NoneType:
            sensor_fusion == env.get_sensor_fusion()
        self.next_pop(env, sensor_fusion, true_dt)
        self.particle_swarm_optimization(env, iterations=iterations,sensor_fusion=sensor_fusion)
        if verbose: print(f"fit={self.global_fit} | {self.global_best[:2]}")
        return translate_differential_drive(*self.global_best[:2])
    
    def shift_ind(self, ind, true_dt):
        # true_dt is the time an action will actually be performed,
        # self.dt is the time of a simulation step
        # this function assumes the first action of ind has been performed for true_dt and shifts the remaining actions to maintain the current trajectory
        sub_vectors = [ind[i:i+self.v_dim] for i in range(0, len(ind), self.v_dim)]
        sub_vectors.append(self.generate_v_vector(horizon=1))
        assert self.dt >= true_dt
        left_fac = (self.dt - true_dt) / self.dt
        right_fac = (true_dt) / self.dt
        for i in range(len(sub_vectors)-1):
            sub_vectors[i] = sub_vectors[i] * left_fac + sub_vectors[i+1] * right_fac  
        ind = np.array(sub_vectors[:-1]).flatten()
        return ind
    
    def next_pop(self, env: Environment, sensor_fusion, true_dt):
        for i in range(len(self.pop)):
            self.pop[i] = self.shift_ind(self.pop[i], true_dt)
        self.charged_pop = self.gen_swarm()
        self.velocities = [np.zeros_like(ind) for ind in self.pop]
        self.individual_best = list(self.pop)
        self.individual_fit = [self.fitness(ind, env, sensor_fusion) for ind in self.individual_best]
        self.global_best = self.individual_best[np.argmin(self.individual_fit)]
        self.global_fit = np.min(self.individual_fit)
    
    def generate_v_vector(self, horizon=0):
        if horizon <= 0:
            horizon = self.horizon
        v_range = self.max_v - self.min_v
        vec = np.array([self.min_v]*horizon*self.v_dim) + np.random.rand(horizon*self.v_dim) * v_range
        return vec
    
    def get_trajectory(self, vec, initial_state):
        positions = [initial_state]
        for v,w in translate_vector(vec):
            positions.append(move_turtle(positions[-1], v, w, self.dt))
        return positions

    def fitness(self, vec, env:Environment, sensor_fusion=None, discount=0.98):
        trans_v_list = translate_vector(vec)
        positions = []
        cur_state = env.get_internal_state()
        if type(sensor_fusion) == NoneType:
            sensor_fusion = env.get_sensor_fusion()
        overall_fit = 0
        for i, (v, w) in enumerate(trans_v_list):
            cur_state = move_turtle(cur_state, v, w, self.dt)
            positions.append(cur_state)
            fit = env.fitness_single(cur_state, sensor_fusion=sensor_fusion, v=np.array([v,w]))
            if fit >= env.collision_penalty:
                overall_fit += env.collision_penalty * (len(trans_v_list) - i)
                break
            else: 
                overall_fit += fit*(discount**i)
        return overall_fit

    def gen_swarm(self):
        return [self.generate_v_vector(self.horizon) for _ in range(self.samples)] # + [np.array([self.max_v]*self.horizon*2)]

    def apply_swarm_forces(self, ind, ind_best, global_best, velocity, inertia=1.3, cog=1.5, soc=1.5):
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

    def apply_charged_forces(self, c_pop, pop, d):
        new_c_pop = []
        for i, x1 in enumerate(c_pop):
            total_force = np.zeros_like(x1)
            for j, x2 in enumerate(c_pop + pop):
                if i != j:
                    total_force += function_3(x1, x2, d)
            new_ind = np.array([max(min(i ,self.max_v), self.min_v) for i in x1 + total_force / (len(c_pop + pop) - 1)])
            new_c_pop.append(new_ind)
        return new_c_pop

    def particle_swarm_optimization(self, env:Environment, iterations=100, sensor_fusion=None, turbulence=0.5):
        start = time()
        # ATTENTION!: fitness is minimized
        if sensor_fusion == None:
            sensor_fusion = env.get_sensor_fusion()
        fit = lambda ind: self.fitness(ind, env, sensor_fusion=sensor_fusion)
        # TODO: make compatible with charged particles
        for iteration in range(iterations):
            new_pop = []
            self.charged_pop = self.apply_charged_forces(self.charged_pop, self.pop, d=10)
            for vec in self.charged_pop:
                f = fit(vec)
                if f < self.global_fit:
                    self.global_fit = f
                    self.global_best = vec
            for i,ind in enumerate(self.pop):
                if turbulence >= np.linalg.norm(self.velocities[i]) and iteration > 0:
                    new_ind = self.generate_v_vector(self.horizon)
                else:
                    new_ind, self.velocities[i] = self.apply_swarm_forces(ind, self.individual_best[i], self.global_best, self.velocities[i])
                new_pop.append(new_ind)
                new_fit = fit(new_ind)
                if new_fit < self.individual_fit[i]:
                    self.individual_fit[i] = new_fit
                    self.individual_best[i] = new_ind
                    if new_fit < self.global_fit:
                        self.global_fit = new_fit
                        self.global_best = new_ind
            self.pop = new_pop
        return self.global_best, self.global_fit
    

####### Single PSO ######
class PSO_Controller(Multi_PSO_Controller):

    dist_koeff = -500
    heading_koeff = 10
    speed_koeff = 1
    comfort_dist = 2

    def __init__(self, samples=10, max_v=22.2, min_v=-22.2, dt=0.05, horizon = 20) -> NoneType:
        super().__init__(samples, max_v, min_v, 1, dt)
        self.lookahead_horizon = horizon

    def next_pop(self, env: Environment, sensor_fusion, true_dt):
        self.pop = self.gen_swarm()
        self.pop[0] = self.global_best
        self.charged_pop = self.gen_swarm()
        self.velocities = [np.zeros_like(ind) for ind in self.pop]
        self.individual_best = list(self.pop)
        self.individual_fit = [self.fitness(ind, env, sensor_fusion) for ind in self.individual_best]
        self.global_best = self.individual_best[np.argmin(self.individual_fit)]
        self.global_fit = np.min(self.individual_fit)
    
    def fitness(self, vec, env:Environment, sensor_fusion=None):
        cur_state = env.get_internal_state()
        v,w = translate_differential_drive(*vec)
        if type(sensor_fusion) == NoneType:
            sensor_fusion = env.get_sensor_fusion()
        next_state = move_turtle(cur_state, v, w, self.dt * self.lookahead_horizon)

        if not sensor_fusion.is_empty:
            pos_point = shapely.Point(next_state[:2])
            dist = (pos_point.distance(sensor_fusion) / env.robo_radius) 
            if dist <= 1:
                return - np.inf
            dist_fit = (1 - sigmoid((dist - self.comfort_dist / 2) * 4 / self.comfort_dist)) * self.dist_koeff
        else:
            dist_fit = 0

        goal_vec = env.goal_pos - next_state[:2]
        heading_vec = move_turtle(next_state, 10, 0, 1) - next_state
        #print("vecs:", goal_vec, heading_vec)
        heading_fit = (goal_vec @ heading_vec[:2] / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec)) * self.heading_koeff #  * np.sign(v)

        speed_fit = v / self.max_v * self.speed_koeff

        #print(f"({v}, {w}):\n{dist_fit}\n{heading_fit}\n{speed_fit}")

        return -(dist_fit + heading_fit + speed_fit)

