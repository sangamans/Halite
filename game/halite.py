from typing import List
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from game import Action, Game

# kaggle halite environment
class Halite(Game):
    def __init__(self, discount):
        super().__init__(discount)
        self.env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
        self.agent_count = 4
        self.actions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, ShipAction.CONVERT, ShipyardAction.SPAWN, None]
        self.obervations = [self.env.reset(self.agent_count)]
        self.done = False

    # to return the number of actions
    def action_space_size(self):
        return len(self.actions)

    # execute one step of the game conditioned by given action to return reward at each step
    def step(self, action):
        self.env.reset(self.agent_count)
        #observation = 
        #self.obervations += 
