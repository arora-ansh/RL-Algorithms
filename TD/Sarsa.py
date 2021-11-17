# SARSA is an on policy TD control algorithm.
# The basic formula for the algorithm is:
# Q(s,a) = Q(s,a) + alpha * (R(s,a) + gamma * Q(s',a') - Q(s,a))
# This is very similar to the one used in TD[0], with the difference of action value being taken. The action is selected in an epsilon greedy manner.

# We will initialize a learning rate, gamma, and a threshold for convergence. 
# We will also initialize the Q-table, which will be a dictionary of dictionaries. We will initialize it arbitrarily except Q(terminal state, any action) = 0.
# Then we will run episodes where we start of with a random state, and then we will select an action based on the policy, E-greedily. 
# We will then observe R and S' based on the S and A taken, and select an A' from S' using E-greedy policy from Q. 
# Having S A R S' A', we perform the table update step and repeat this for that episode until we reach the end state for that episode.

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

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    # Initialize Q-table of zeros
    Q = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            Q[(s,a)] = 0.0

    # Initialize variables
    iterations = []
    rewards = []

    # We run episodes 
    for i in range(num_episodes):
        
        total_reward_ep = 0
        total_iteration_ep = 0

        state = env.reset() # Initialize the state of environment arbitrarily
        action = epsilon_greedy_policy(Q, state, epsilon) # Initialize the action taken based on the policy
        reward = 0 # Initialize the reward
        done = False # Initialize the done flag

        # Loop until the episode is done
        while not done:
            total_iteration_ep += 1 # Increment the number of iterations
            next_state, reward, done, _ = env.step(action) # Take the action and get the next state (S'), reward(R), and done flag
            next_action = epsilon_greedy_policy(Q, next_state, epsilon) # Select an action based on epsilon-greedy policy
            # Update the Q-table
            Q[(state,action)] += alpha * (reward + gamma * Q[(next_state,next_action)] - Q[(state,action)])
            state = next_state # Update the state
            action = next_action # Update the action
            total_reward_ep += reward # Update the total reward earned
        
        rewards.append(total_reward_ep) # Append the total reward earned for this episode
        iterations.append(total_iteration_ep) # Append the total number of iterations for this episode

    return Q, rewards, iterations

Q, rewards, iterations = sarsa(env, num_episodes=1000, alpha=0.4, gamma=0.999, epsilon=0.05)

# Plotting Iterations taken and Rewards gotten at each episode
plt.figure()
plt.plot(iterations, label = 'Iterations')
plt.plot(rewards, label = 'Rewards')
plt.xlabel('Episode')
plt.ylabel('Rewards/Iterations')
plt.legend()
plt.show()
