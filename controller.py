from environment import Environment
from typing import Tuple


class Controller():

    def __call__(self, ENV:Environment, dt:float) -> Tuple[float, float]:
        return 0.0, 0.0

class PlayerController(Controller):
    # TODO: Player Controll
    pass

class AnimationController(Controller):
    # TODO: Animation Controller
    pass