from environment import Environment
import numpy as np
from Turtlebot_Kinematics import KinematicModel, unicycleKin
from time import time
from functions import function_3, sigmoid, comf_distance
import shapely
from controller import Controller

def timer(func):
    def inner(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(func.__name__, time() - start)
        return result
    return inner

def ind_to_vectorlist(ind):
    out = []
    for i in range(0,len(ind), 2):
        sub_vec = tuple(ind[i:i+2])
        out.append(sub_vec)
    return out

def disturb(vec):
    return vec * np.random.rand(len(vec))


####### Multi PSO ######
class Multi_PSO_Controller(Controller):
    v_dim = 2
    def __init__(self, samples=10, kinematic: KinematicModel = None, horizon=10, dt=0.05, iterations=10) -> None:
        self.samples = samples
        self.kinematic = kinematic if not kinematic is None else unicycleKin()
        self.horizon = horizon
        self.dt = dt
        self.iterations = iterations

        self.pop = self.gen_swarm()
        self.charged_pop = self.gen_swarm()
        self.velocities = [np.zeros_like(ind) for ind in self.pop]
        self.individual_best = list(self.pop)
        self.individual_fit = [np.infty for _ in self.individual_best]
        self.global_best = self.individual_best[np.argmin(self.individual_fit)]
        self.global_fit = np.min(self.individual_fit)

    @timer
    def __call__(self, env: Environment, true_dt=0.05, sensor_fusion=None, verbose=True):
        if sensor_fusion is None:
            sensor_fusion == env.get_sensor_fusion()
        self.next_pop(env, sensor_fusion, true_dt)
        self.particle_swarm_optimization(env, iterations=self.iterations,sensor_fusion=sensor_fusion)
        if verbose: print(f"fit={self.global_fit} | {self.global_best[:2]}")
        return self.global_best[0], self.global_best[1]
    
    def shift_ind(self, ind, true_dt):
        # true_dt is the time an action will actually be performed,
        # self.dt is the time of a simulation step
        # this function assumes the first action of ind has been performed for true_dt and shifts the remaining actions to maintain the current trajectory
        sub_vectors = [ind[i:i+self.v_dim] for i in range(0, len(ind), self.v_dim)]
        sub_vectors.append(self.kinematic.generate_v_vector(horizon=1))
        assert self.dt >= true_dt
        left_fac = (self.dt - true_dt) / self.dt
        right_fac = (true_dt) / self.dt
        for i in range(len(sub_vectors)-1):
            sub_vectors[i] = sub_vectors[i] * left_fac + sub_vectors[i+1] * right_fac  
        ind = np.array(sub_vectors[:-1]).flatten()
        if -50 >= min(ind) or max(ind) >= 50:
                raise Exception(f"velocities not snapped correctly ({ind})")
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
    
    def get_trajectory(self, vec, initial_state):
        positions = [initial_state]
        for v,w in ind_to_vectorlist(vec):
            positions.append(self.kinematic(positions[-1], v, w, self.dt))
        return positions

    def fitness(self, vec, env:Environment, sensor_fusion=None, discount=0.98):
        trans_v_list = ind_to_vectorlist(vec)
        positions = []
        cur_state = env.get_internal_state()
        if sensor_fusion is None:
            sensor_fusion = env.get_sensor_fusion()
        overall_fit = 0
        for i, (v, w) in enumerate(trans_v_list):
            cur_state = self.kinematic(cur_state, v, w, self.dt)
            positions.append(cur_state)
            fit = env.fitness_single(cur_state, sensor_fusion=sensor_fusion, v=np.array([v,w]))
            if fit >= env.collision_penalty:
                overall_fit += env.collision_penalty * (len(trans_v_list) - i)
                break
            else: 
                overall_fit += fit*(discount**i)
        return overall_fit

    def gen_swarm(self):
        return [self.kinematic.generate_v_vector(self.horizon) for _ in range(self.samples)]

    def apply_swarm_forces(self, ind, ind_best, global_best, velocity, inertia=1.3, cog=1.5, soc=1.5):
        new_velocity = velocity * inertia + cog * disturb(ind_best - ind) + soc * disturb(global_best - ind) 
        new_ind = ind + new_velocity
        new_ind, new_velocity = self.kinematic.snap_velocities(new_ind, new_velocity)
        if -50 >= min(new_ind) or max(new_ind) >= 50:
                raise Exception(f"velocities not snapped correctly ({new_ind})")
        return new_ind, new_velocity

    def apply_charged_forces(self, c_pop, pop, d):
        new_c_pop = []
        avg_dists = []
        for i, x1 in enumerate(c_pop):
            dists = []
            total_force = np.zeros_like(x1)
            for j, x2 in enumerate(c_pop + pop):
                if i != j:
                    total_force += comf_distance(x1, x2, d, k=0.1) #function_3(x1, x2, d)
                    dists.append(np.linalg.norm(x2 - x1))
            new_ind = x1 + total_force # / (len(c_pop + pop) - 1))
            new_ind, _ = self.kinematic.snap_velocities(new_ind)
            if -50 >= min(new_ind) or max(new_ind) >= 50:
                raise Exception(f"velocities not snapped correctly ({new_ind})")
            new_c_pop.append(new_ind)
            avg_dists.append(sum(dists) / len(dists))
        print("avg charged dist:", sum(avg_dists) / len(avg_dists))
        return new_c_pop

    def particle_swarm_optimization(self, env:Environment, iterations=100, sensor_fusion=None, turbulence=0.5):
        start = time()
        # ATTENTION!: fitness is minimized
        if sensor_fusion == None:
            sensor_fusion = env.get_sensor_fusion()
        fit = lambda ind: self.fitness(ind, env, sensor_fusion=sensor_fusion)
        for iteration in range(iterations):
            new_pop = []
            self.charged_pop = self.apply_charged_forces(self.charged_pop, self.pop, d=2)
            for vec in self.charged_pop:
                f = fit(vec)
                if f < self.global_fit:
                    self.global_fit = f
                    self.global_best = vec
            for i,ind in enumerate(self.pop):
                if turbulence >= np.linalg.norm(self.velocities[i]) and iteration > 0:
                    new_ind = self.kinematic.generate_v_vector(self.horizon)
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

    def __init__(self, samples=10, kinematic: KinematicModel = None, dt=0.5) -> None:
        super().__init__(samples, KinematicModel, 1, dt)

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
        v1,v2 = vec[0], vec[1]
        if sensor_fusion is None:
            sensor_fusion = env.get_sensor_fusion()
        next_state = self.kinematic(cur_state, v1, v2, self.dt)

        if not sensor_fusion.is_empty:
            pos_point = shapely.Point(next_state[:2])
            dist = (pos_point.distance(sensor_fusion) / env.robo_radius) 
            if dist <= 1:
                return - np.inf
            dist_fit = (1 - sigmoid((dist - self.comfort_dist / 2) * 4 / self.comfort_dist)) * self.dist_koeff
        else:
            dist_fit = 0

        goal_vec = env.goal_pos - next_state[:2]
        heading_vec = self.kinematic.heading(next_state, v1, v2)
        #print("vecs:", goal_vec, heading_vec)
        heading_fit = (goal_vec @ heading_vec[:2] / np.linalg.norm(goal_vec) / np.linalg.norm(heading_vec)) * self.heading_koeff * np.sign(v1)

        speed_fit = self.kinematic.relativ_speed * self.speed_koeff

        #print(f"({v}, {w}):\n{dist_fit}\n{heading_fit}\n{speed_fit}")

        return -(dist_fit + heading_fit + speed_fit)

