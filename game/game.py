# helper functions
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from typing import List
from abc import abstractmethod, ABC

# actions of the game
class Action(object):
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

# can be used when adding more players
class Player(object):
    pass

# keeps track of the actions
class ActionHistory(object):
    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()

# abstract class to implement the halite environment
class Game(object):
    def __init__(self, discount: float):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    # applying actions to the environment
    def apply(self, action: Action):
        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    # storing stats of each MCTS (Monte Carlo Tree Search) run
    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    # targets to learn from during the network training
    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    # return the current player
    def to_play(self) -> Player:
        return Player()

    # action history of actions executed
    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)