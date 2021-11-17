# Double Q-Learning, which makes use of two separate Q-values to select the action

import gym 
import random 
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

def epsilon_greedy_policy(Q, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample() # Select a random action with probability epsilon
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)]) # Select the action with the highest Q-value

def double_q_learning(env, num_episodes, gamma = 1.0, alpha = 0.5, epsilon = 0.1):
    # Initialize Q1-table of zeros
    Q1 = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            Q1[(s,a)] = 0.0
    # Initialize Q2-table of zeros
    Q2 = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            Q2[(s,a)] = 0.0

    # Initialize variables
    iterations = []
    rewards = []

    # We run episodes
    for i in range(num_episodes):
        total_reward_ep = 0
        total_iteration_ep = 0

        state = env.reset()
        reward = 0 # Initialize the reward
        done = False # Initialize the done flag

        # Loop until the episode is done
        while not done:
            action = epsilon_greedy_policy(Q1, state, epsilon) # Select an action
            next_state, reward, done, _ = env.step(action) # Take the action
            total_reward_ep += reward # Update the total reward
            total_iteration_ep += 1 # Increment the number of iterations
            # Select an action
            action = epsilon_greedy_policy(Q2, next_state, epsilon)
            # Update Q1
            Q1[(state,action)] = Q1[(state,action)] + alpha * (reward + gamma * Q2[(next_state,action)] - Q1[(state,action)])
            # Update state
            state = next_state