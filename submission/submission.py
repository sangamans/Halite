from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from random import choice
import json

# Writing to json to see how the data gets outputted
def writetojson(filename, data):
    filename = filename + ".json"
    with open(filename, 'w') as fp:
        json.dump(data, fp)

# Create environment
env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
agent_count = 4
print(env.configuration)

# creating the agent that will act random choice
def random_agent(obs,config):
    
    board = Board(obs,config)
    me = board.current_player
    
    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT,None])
    
    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = choice([ShipyardAction.SPAWN,None])
    
    return me.next_actions

env.reset(agent_count)
#env.run([random_agent, "random", "random", "random"])
#observation = env.render(mode="json")
#data = json.loads(observation)
#writetojson("test", data)
random_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT,None])
data = env.step([ShipAction.NORTH, random_action, random_action, random_action])
print(data[1]['observation'])