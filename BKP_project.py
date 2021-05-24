import or_gym
import numpy as np
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
import importlib

# moduleName = input('model v3py.py')
# importlib.import_module(moduleName)
# moduleName = input('DQN v1.py')
# importlib.import_module(moduleName)

# In this code, you will fin a description of the environment including state, actions, and rewards

env = or_gym.make('Knapsack-v2')
print(env)
env.mask = False
env.reset()

action_space=env.action_space.n #Get number of actions in an environment 
print('Action space', action_space)
state_space=env.state #Get number of states in an environment 
print('State space', state_space)

print('One episode is finished when total weight is <=',env.max_weight)
print('Agent receives a reward named value, which belongs to that weight')
print('Knapsack-v2 is considered solved when the agent obtains an average reward of 1600 (or above) over 100 consecutive episodes.')

action=env.action_space.sample()# sample an action from the env instanceprint(action)
print(env.step(action), 'with a probability of 1/200' )# call the step function of the env. and inspect quadruple of (state, reward, done, info)

for episode in range(2):
    state = env.reset()# get the starting state from the env.
    step = 0
    done = False
    print("EPISODE ", episode)
    for step in range(99):
        action = env.action_space.sample()# sample an action from the environment
        new_state, reward, done, info = env.step(action) #give the action to environment to obtain reward, and next state,
        if done: #if the goal state is reached or agent fall into hole.
            new_state == 200
            print("We reached our Goal 🏆")

# We print the number of step it took.
            print("Number of items selected", step) 
            #The number of actions is equal to the number of items available.

            break
        state = new_state
env.close()
