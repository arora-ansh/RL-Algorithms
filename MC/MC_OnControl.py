# In Control of MC, we use MC Prediction to figure out the State Values (Policy Evaluation), and then use these to figure out the Policy (Policy Improvement).
# This is an extension to The Generalized Policy Iteration Algorithm, where here in this case we just use MC to get the State Values.

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v1')

# On Policy First Visit MC Control (for E-Soft Policies) to estimate Policy 
def MC_OnControl(env, num_episodes, gamma=1.0, epsilon=0.1):
    # Initiate Random E-Soft Policy

    # Initialize empty dictionaries for Q-table 
    Q = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            Q[(s,a)] = 0.0
    
    # Initialize Returns
    Returns = {}
    
    
