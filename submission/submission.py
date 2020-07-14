from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from random import choice

# Create environment
env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
print(env.configuration)

# creating the agent
def agent(obs,config):
    
    board = Board(obs,config)
    me = board.current_player
    
    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,None])
    
    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None
    
    return me.next_actions

