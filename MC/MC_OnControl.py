# In Control of MC, we use MC Prediction to figure out the State Values (Policy Evaluation), and then use these to figure out the Policy (Policy Improvement).
# This is an extension to The Generalized Policy Iteration Algorithm, where here in this case we just use MC to get the State Values.

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v1')
