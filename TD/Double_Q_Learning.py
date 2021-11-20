# Double Q-Learning, which makes use of two separate Q-values to select the action

import gym 
import random 
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

# Epsilon Greedy Policy for Q1 and Q2 combined
def dq_epsilon_greedy_policy(Q1, Q2, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample() # Select a random action with probability epsilon
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q1[(state,x)] + Q2[(state,x)]) # Select the action with the highest Q-value


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
            action = dq_epsilon_greedy_policy(Q1, Q2, state, epsilon) # Select an action
            next_state, reward, done, _ = env.step(action) # Take the action
            total_reward_ep += reward # Update the total reward
            total_iteration_ep += 1 # Increment the number of iterations
            # Update the Q-table using Double Q-Learning, Q1 with the action from Q2 with 0.5 probability and Q2 with the action from Q1 with 0.5 probability
            if random.random() < 0.5:
                # Action is selected for Q1 from Q2 with argmax
                Q1[(state, action)] = Q1[(state, action)] + alpha * (reward + gamma * max(list(range(env.action_space.n)), key = lambda x: Q2[(next_state,x)]) - Q1[(state, action)])
            else:
                # Action is selected for Q2 from Q1 with argmax
                Q2[(state, action)] = Q2[(state, action)] + alpha * (reward + gamma * max(list(range(env.action_space.n)), key = lambda x: Q1[(next_state,x)]) - Q2[(state, action)])
            # Update state
            state = next_state
        
        rewards.append(total_reward_ep) # Append the total reward earned for this episode
        iterations.append(total_iteration_ep) # Append the total number of iterations for this episode

    return Q1, Q2, rewards, iterations

Q1, Q2, rewards, iterations = double_q_learning(env, num_episodes = 1000, gamma = 0.999, alpha = 0.4, epsilon = 0.05)

# Plotting Iterations taken and Rewards gotten at each episode
plt.figure()
plt.plot(iterations, label = 'Iterations')
plt.plot(rewards, label = 'Rewards')
plt.xlabel('Episode')
plt.ylabel('Rewards/Iterations')
plt.legend()
plt.show()
