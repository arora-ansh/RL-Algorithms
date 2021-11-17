import gym 
import numpy as np 
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

# Here we use the policy iteration algorithm to solve the Frozen Lake problem. 
# We perform policy evaluation and policy improvement in each iteration. 

def get_v(s, a, V, gamma):
    v = 0
    for prob, next_state, reward, done in env.P[s][a]:
        v += prob * (reward + gamma * V[next_state]) # p(s',r|s,a) * (r + gamma * V(s'))
    return v

def policy_evaluation(env, policy, gamma = 1.0, theta = 1e-10, max_iteration = 100000):
    V = np.zeros(env.observation_space.n) # initialize value function to value 0 arbitrarily
    for i in range(max_iteration):
        delta = 0 # The change in V(s) values observed between last and this iteration
        for s in range(env.observation_space.n): # Iterarate over all states
            v = V[s] #Last value of V(s)
            V[s] = get_v(s, policy[s], V, gamma)
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(env, V, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        policy[s] = np.argmax([get_v(s, a, V, gamma) for a in range(env.action_space.n)])
    return policy

def policy_iteration(env, gamma = 1.0, theta = 1e-10, max_iteration = 100000):
    policy = np.zeros(env.observation_space.n) # initialize policy to value 0 arbitrarily
    V = policy_evaluation(env, policy, gamma, theta, max_iteration)
    while True:
        V = policy_evaluation(env, policy, gamma, theta, max_iteration)
        policy_stable = True
        new_policy = policy_improvement(env, V, gamma)
        for s in range(env.observation_space.n):
            if policy[s] != new_policy[s]:
                policy_stable = False
        if policy_stable:
            policy = new_policy
            break
        policy = new_policy

    return policy, V

optimal_policy, optimal_value_function = policy_iteration(env)
print("Optimal policy:", optimal_policy)
print("Optimal value function:", optimal_value_function)
