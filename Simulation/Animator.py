import pygame
from environment import Environment


class Animator:
    @staticmethod
    def from_contoller(controller):
        pass

    def from_player():
        pass

    def from_replay(replay):
        pass

    def __init__(self, map: Environment, meta_data = None, screen_size = (1280,720)):
        self.map = map
        self.dt = meta_data["dt"] if "dt" in meta_data else 1 / 20
        self.screen = pygame.display.set_mode(screen_size)

    def start(self):
        pygame.init()

    def end(self):
        pygame.quit()

    def next(self, robo_pos, goal_pos):
        pass