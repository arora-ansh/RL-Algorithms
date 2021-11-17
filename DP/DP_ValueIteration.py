import gym 
import numpy as np 
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

def get_v(s, a, V, gamma):
    v = 0
    for prob, next_state, reward, done in env.P[s][a]:
        v += prob * (reward + gamma * V[next_state]) # p(s',r|s,a) * (r + gamma * V(s'))
    return v

def value_iteration(env, gamma = 1.0, threshold = 1e-10, max_iteration = 100000):
    V = np.zeros(env.observation_space.n) # initialize value function to value 0 arbitrarily

    # Loop over the number of iterations
    for i in range(max_iteration):
        delta = 0 # The change in V(s) values observed between last and this iteration
        for s in range(env.observation_space.n): # Iterarate over all states
            v = V[s] #Last value of V(s)
            V[s] = np.max([get_v(s, a, V, gamma) for a in range(env.action_space.n)]) # Update the value of V(s) using max over all actions
            delta = max(delta, np.abs(v - V[s])) # Update delta as the change in V(s) values observed between last and this iteration
        if delta < threshold: #Stop the loop if the change in V(s) values observed between last and this iteration is less than the threshold
            break
    return V

optimal_value_function = value_iteration(env)

def get_policy(env, V, gamma = 1.0):
    policy = np.zeros(env.observation_space.n) # initialize policy to value 0 arbitrarily
    for s in range(env.observation_space.n): # Iterate over all states, we'll find each state's optimal action
        policy[s] = np.argmax([get_v(s, a, V, gamma) for a in range(env.action_space.n)]) # Find the optimal action for each state using argmax
    return policy

optimal_policy = get_policy(env, optimal_value_function)

print(optimal_value_function)
print(optimal_policy)