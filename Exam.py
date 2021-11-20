from random import randint
import matplotlib.pyplot as plt
import numpy as np

def epsilon_greedy_action(Q, state, epsilon):
    if np.random.random() < epsilon:
        return randint(0,1) # Explore action space with epislon probability
    else:
        if Q[state,0] == Q[state,1]:
            return randint(0,1)
        return np.argmax(Q[state])

def dq_epsilon_greedy_policy(Q1, Q2, state, epsilon):
    if np.random.random() < epsilon:
        return randint(0,1)# Select a random action with probability epsilon
    else:
        if Q1[state,0]+Q2[state,0] == Q1[state,1]+Q2[state,1]:
            return randint(0,1)
        return np.argmax(Q1[state]+Q2[state])

def double_q_learning(gamma, alpha, num_episodes, epsilon, max_steps, twist = False, twist_eps = 0.1):
    # Initialize Q1 Action Value table of zeros
    Q1 = np.zeros((6,2)) # Since there are 6 states and 2 actions per state 
    # Initialize Q2 Action Value table of zeros
    Q2 = np.zeros((6,2))

    # Initialize variables
    iterations = []
    rewards = []

    # We run the episodes
    for i in range(num_episodes):
        total_reward_ep = 0
        total_iteration_ep = 0

        state = randint(0,5) # Initialize state, acts as environment reset
        reward = 0 # Initialize the reward
        num_steps = 0

        while num_steps < max_steps:
            # Choose an action based on epsilon greedy policy from Q1 + Q2, 0 is left 1 is right
            action = dq_epsilon_greedy_policy(Q1, Q2, state, epsilon)

            if action == 0:
                if state == 0:
                    next_state = 0
                else:
                    next_state = state - 1
                reward = 0
            else:
                if state == 5:
                    next_state = 5
                else:
                    next_state = state + 1
                if next_state < 5:
                    reward = 1
                else:
                    # We take a random Gaussian distribution with mean -5 and standard deviation 3
                    reward = np.random.normal(-5,3)
            
            # print(state,action,next_state,reward)
            
            # We have both our next state and our reward, we will now update our Q values
            # Update Q1 and Q2
            if not twist:
                if np.random.random() < 0.5:
                    Q1[state, action] = Q1[state, action] + alpha*(reward + gamma*Q2[next_state,np.argmax(Q1[next_state, :])] - Q1[state, action])
                else:
                    Q2[state, action] = Q2[state, action] + alpha*(reward + gamma*Q1[next_state,np.argmax(Q2[next_state, :])] - Q2[state, action])
            else: # DQLT
                if np.random.random() < 0.5:
                    if np.random.random() < twist_eps:
                        Q1[state, action] = Q1[state, action] + alpha*(reward + gamma*Q2[next_state,randint(0,1)] - Q1[state, action])
                    else:
                        Q1[state, action] = Q1[state, action] + alpha*(reward + gamma*Q2[next_state,np.argmax(Q1[next_state, :])] - Q1[state, action])
                else:
                    Q2[state, action] = Q2[state, action] + alpha*(reward + gamma*Q1[next_state,np.argmax(Q2[next_state, :])] - Q2[state, action])

            # Update state and reward
            state = next_state
            total_reward_ep += reward
            total_iteration_ep += 1
            num_steps += 1
        
        iterations.append(total_iteration_ep)
        rewards.append(total_reward_ep)
        # print()

    return Q1, Q2, iterations, rewards

Q1, Q2, iterations, rewards = double_q_learning(gamma = 0.98, alpha = 0.1, num_episodes = 1000, epsilon = 0.05, max_steps = 200)
Q1, Q2, iterations, rewards_dqlt = double_q_learning(gamma = 0.98, alpha = 0.1, num_episodes = 1000, epsilon = 0.05, max_steps = 200, twist = True, twist_eps=0.05)

plt.figure()
# plt.plot(iterations, label = 'Iterations')
plt.plot(rewards, label = 'DQL')
plt.plot(rewards_dqlt, '-', label = 'DQLT')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend()
plt.show()