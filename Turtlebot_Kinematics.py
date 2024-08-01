import numpy as np
from dataclasses import dataclass
from typing import Tuple, Generator, Any
from numpy.core.multiarray import array as array
import pandas

a = np.array([1,0])
def rotation_matrix(angle):
    return np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])

def rotate(vec: np.ndarray, angle: float):
    return rotation_matrix(angle) @ vec

# unicycle model
def move_turtle(state: np.array, v:float, w:float, t:float) -> np.array:
    if len(state) < 3:
        raise Exception(f"state expected to have 3 elements. recieved: {state}")
    x, y, theta = state
    if w != 0:
        r = v / w
        wt = w * t
        return np.array([
            x + r * np.sin(wt) * np.cos(theta) + (r - r * np.cos(wt)) * np.sin(theta),
            y + r * np.sin(wt)*(-np.sin(theta))+ (r - r * np.cos(wt)) * np.cos(theta),
            theta + wt
        ])
    else:
        return np.array([
            x + np.cos(theta) * v * t,
            y - np.sin(theta) * v * t,
            theta
        ])
    
# translating unicycle model
def translate_differential_drive(v_left: float, v_right: float, wheel_distance:float = 16.0):
    # based on Kinematic model described by Nele Traichel in her Master Thesis
    trans = (v_right + v_left) / 2
    rotation = (v_right - v_left) / wheel_distance
    # returns translational and rotational speed of robot that results from the input velocities
    return trans, rotation


### Kinematic Functions ###
# signature: state: np.array, v1, v2, t (floats) -> np.array (new array)
class KinematicModel():
    v1_max: float
    v1_min: float
    v2_max: float
    v2_min: float

    def __call__(self, state: np.array, v1:float, v2:float, t:float) -> np.array:
        # returns new state after applying v1 and v2 for t seconds in Kinematic model
        pass

    def clone(self):
        return type(self)()

    def v_spaces(self, samples) -> Tuple[np.linspace, np.linspace]:
        # util for creating linspaces of possible velocities
        return np.linspace(self.v1_min, self.v1_max, samples), np.linspace(self.v2_min, self.v2_max, samples)
    
    def v_gen(self, samples) -> Generator[Tuple[np.array, np.array], Any, Any]:
        # generator over possible volocities
        space1, space2 = self.v_spaces(samples)
        for v1 in space1:
            for v2 in space2:
                yield v1, v2

    def relativ_speed(self, v1, v2) -> float:
        pass

    def snap_velocities(self, ind, velocities = None) -> Tuple[np.array, np.array]:
        if velocities is None:
            velocities = np.zeros_like(ind)
        # ensures ind's velocities are within the limits and sets velocities 0 if they where outside possible range
        for i in range(0, len(ind), 2):
            if ind[i] < self.v1_min:
                ind[i] = self.v1_min
                velocities[i] = 0
            elif ind[i] > self.v1_max:
                ind[i] = self.v1_max
                velocities[i] = 0
            if ind[i+1] < self.v2_min:
                ind[i+1] = self.v2_min
                velocities[i+1] = 0
            elif ind[i+1] > self.v2_max:
                ind[i+1] = self.v2_max
                velocities[i+1] = 0
            assert self.v1_min <= ind[i] <= self.v1_max and self.v2_min <= ind[i+1] <= self.v2_max
        return ind, velocities
    
    # for models that care about agents direction
    def heading(self, state: np.array, v1:float, v2:float) -> np.array:
        return move_turtle(state, 10, 0, 0.1) - state
    
    def generate_v_vector(self, horizon) -> np.array:
        li = []
        v1_range = self.v1_max - self.v1_min
        v2_range = self.v2_max - self.v2_min
        for _ in range(horizon):
            li.append(self.v1_min + np.random.rand() * v1_range) 
            li.append(self.v2_min + np.random.rand() * v2_range) 
        assert max(li) <= max(self.v1_max, self.v2_max)
        return np.array(li)
    
    def max_speed(self):
        return self.v1_max

@dataclass
class unicycleKin(KinematicModel):
    # velovity limits derived from differential drive of v_left = 22.2, v_right = 8.0
    # translational speed
    v1_min: float = -15
    v1_max: float = 15
    # rotational speed
    v2_min: float = -0.9
    v2_max: float = 0.9
    def __call__(self, state: np.array, v:float, w:float, t:float) -> np.array:
        return move_turtle(state, v, w, t)
    
    def relativ_speed(self, v1, v2) -> float:
        return v1 / self.v1_max

@dataclass
class difDriveKin(KinematicModel):
    # left speed
    v1_min: float = -22.2
    v1_max: float = 22.2
    # right speed
    v2_min: float = -22.2
    v2_max: float = 22.2

    def __call__(self, state: np.array, v_left: float, v_right: float, t:float) -> np.array:
        v, w = translate_differential_drive(v_left, v_right)
        return move_turtle(state, v, w, t)
    
    def relativ_speed(self, v1, v2) -> float:
        v,_ = translate_differential_drive(v1, v2)
        return v / self.v1_max

@dataclass
class droneKin(KinematicModel):
    # maximum movement in any direction
    max_v: float = 22.2
    def __call__(self, state:np.array, vx:float, vy:float, t:float) -> np.array:
        x, y, theta = state
        return np.array([
                x + vx * t,
                y - vy * t,
                theta
            ])
            
    def v_gen(self, samples) -> Generator[Tuple, Any, Any]:
        # r is the length of the velocity vector
        # can be seen as radius of a circle
        for r in np.linspace(self.max_v, 0, samples):
            # w represents heading of vector
            for w in np.linspace(0, 2*np.pi, samples):
                # x, y values trace a circle of radius r
                yield r * np.cos(w), r*np.sin(w)
    
    def snap_velocities(self, ind, velocities) -> Tuple:
        # how long the velocitie vector actually is
        norm = abs(np.linalg.norm(ind))
        if norm > self.max_v:
            # snaps velocity vector to max length, while maintaining direction
            ind = ind / norm * self.max_v
            # velocities set to 0 to match other snap implementations, refraction might actually be better though
            velocities = np.zeros_like(velocities)
            return ind, velocities
        else: 
            return ind, velocities
    
    def relativ_speed(self, v1, v2) -> float:
        return np.linalg.norm(np.array([v1, v2])) / self.max_v
        
    # does not care about agents direction, so heading is purely based on current speed vector
    def heading(self, state: np.array, v1: float, v2: float) -> np.array:
        vec = np.array([v1, v2, 0])
        return vec / np.linalg.norm(vec)
    
    def generate_v_vector(self, horizon) -> np.array:
        li = []
        for _ in range(horizon):
            r = np.random.rand() * self.max_v
            w = np.random.rand() * 2 * np.pi
            li.append(r * np.cos(w))
            li.append(r * np.sin(w))
        return np.array(li)
    
    def max_speed(self):
        return self.max_v
    

class AnimationModel(unicycleKin):
    def __init__(self, df_path: str) -> None:
        self.df = pandas.read_hdf(df_path)
        self.i = 0
        super().__init__()

    def __call__(self, state: np.array, v1:float, v2:float, t:float) -> np.array:
        col = self.df.iloc[self.i]
        self.i += 1
        return col["robo_x"], col["robo_y"], col["robo_deg"]
    
@dataclass
class unicycleAcceleration(unicycleKin):
    # actual velocity limits:
    av1_min: float = -15
    av1_max: float = 15
    av2_min: float = -0.9
    av2_max: float = 0.9
    # v limits are actually acceleration limits for simpler implementation
    v1_min: float = av1_min / 3
    v1_max: float = av1_max / 3
    v2_min: float = av2_min / 1
    v2_max: float = av2_max / 1
    # current speeds:
    v1 = 0
    v2 = 0

    # this is an approximation of actual behavior
    def __call__(self, state: np.array, v:float, w:float, t:float) -> np.array:
        new_v1 = self.v1 + v*t
        new_v2 = self.v2 + w*t
        self.v1 = min(max(new_v1, self.av1_min), self.av1_max)
        self.v2 = min(max(new_v2, self.av2_min), self.av2_max)
        avg_v1 = (new_v1 + self.v1) / 2 
        avg_v2 = (new_v2 + self.v2) / 2 
        self.v1 = new_v1
        self.v2 = new_v2
        return move_turtle(state, avg_v1, avg_v2, t)
    
    def clone(self):
        clone = unicycleAcceleration()
        clone.v1 = self.v1
        clone.v2 = self.v2
        return clone
    
    def relativ_speed(self, v1, v2) -> float:
        return self.v1 / self.av1_max
    
    def max_speed(self):
        return self.av1_max
    
    def end_point(self):
        stop_dt = max(self.v1 / self.v1_max, self.v2 / self.v2_max)
        pass


@dataclass
class doneAcceleration(droneKin):
    # actual maximum movement in any direction
    amax_v: float = 22.2
    # this is acceleration limits
    max_v: float = amax_v / 5
    # current speeds:
    v1 = 0.0
    v2 = 0.0

    # this is an approximation of actual behavior
    def __call__(self, state: np.array, v:float, w:float, t:float) -> np.array:
        assert np.linalg.norm(np.array(v, w)) < self.max_v
        new_vx = self.v1 + v*t
        new_vy = self.v2 + w*t
        vs, _ = self.snap_velocities(np.array([new_vx, new_vy]))
        x, y, theta = state
        new_state = np.array([
                x + self.v1 * t + (v/2)*t**2,
                y - self.v2 * t + (w/2)*t**2,
                theta
            ])
        self.v1 = vs[0]
        self.v2 = vs[1]
        return new_state
    
    def clone(self):
        clone = unicycleAcceleration()
        clone.v1 = self.v1
        clone.v2 = self.v2
        return clone
    
    def relativ_speed(self, v1, v2) -> float:
        return np.linalg.norm(np.array([self.v1, self.v2])) / self.max_v
    
    def heading(self, state: np.array, v1: float, v2: float) -> np.array:
        vec = np.array([self.v1, self.v2, 0])
        return vec / np.linalg.norm(vec)
    
    def max_speed(self):
        return self.av1_max
    
    def end_point(self):
        stop_dt = max(self.v1 / self.v1_max, self.v2 / self.v2_max)
        pass