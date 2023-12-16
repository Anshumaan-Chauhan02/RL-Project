#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import gym
import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Tile Coding

# In[ ]:


### Tile Coding

class hashTable():
  def __init__(self, size):
    self.size = size
    self.dict_size = 0
    self.d = {}

  def size_dict(self):
    return len(self.d)

  def getTileIndex(self, coords):
    if coords in self.d:
      return self.d[coords]

    current_len = self.size_dict()
    if current_len >= self.size:
      return hash(coords) % self.size
    else:
      self.d[coords] = current_len
      return current_len

def get_tile_index(hash_table, num_tilings, state, actions=[]):
  quantized_state = [math.floor(state_i * num_tilings) for state_i in state]
  Tiles = []
  for tiling in range(num_tilings):
    add_factor = tiling * 2
    b = add_factor
    coords = [tiling]
    for q_state_i in quantized_state:
      coords.append((q_state_i + b) // num_tilings)
      b += add_factor
    coords.extend(actions)
    Tiles.append(hash_table.getTileIndex(tuple(coords)))
  return Tiles


# ## Policy Network

# In[ ]:


### Action value Network

class ActionNetwork(nn.Module):
  def __init__(self, num_tilings):
    super().__init__()
    self.fc1 = nn.Linear(num_tilings, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128,1)

  def forward(self, state):
    x = self.fc1(torch.Tensor(state))
    x = self.relu(x)
    x = self.fc2(x)
    return x


# ## Semi Gradient n-step SARSA

# In[ ]:


### Training Loop

def policy(action_network, env, hash_table, num_tilings, state, epsilon=0):
  action_value = []
  for i in range(env.action_space.n):
    action_value.append(action_network(get_tile_index(hash_table, num_tilings, state, [i])))
  best_action = torch.argmax(torch.Tensor(action_value)).item()
  list_of_actions = list(range(env.action_space.n))
  list_of_actions.remove(best_action)
  if torch.rand(1).item() < epsilon:
    action = np.random.choice(list_of_actions)
  else:
    action = best_action
  return action

def policy_loss(G, hash_table, num_tilings, action_network, states, actions, tau):
  return 0.5 * ((G - action_network(get_tile_index(hash_table, num_tilings, states[tau], [actions[tau]])).item())**2)

def n_step_sarsa(action_network, env, hash_table, num_tilings, epsilon, n, gamma, epochs, optimizer):
  R = []
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
  for epoch in range(epochs):
    initial_state, _ = env.reset()
    initial_state = (initial_state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
    action = policy(action_network, env, hash_table, num_tilings, initial_state, epsilon)
    T = np.inf
    t = 0
    states = [initial_state]
    actions = [action]
    rewards = []
    epsilon_val = np.exp(0.1*epoch)/3 # epsilon decay 
    while True:
      if t < T:
        next_state, reward, terminate, truncate, _ = env.step(action)
        next_state = (next_state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
        states.append(next_state)
        rewards.append(reward)
        if terminate or truncate:
          T = t + 1
        else:
          action = policy(action_network, env, hash_table, num_tilings, next_state, epsilon_val)
          actions.append(action)
      tau = t - n + 1
      if tau >= 0:
        G = np.sum([(gamma**(i - tau - 1)) * (rewards[i]) for i in range(tau + 1, min(tau + n, T))])
        if tau + n < T:
          G += (gamma**n) * action_network(get_tile_index(hash_table, num_tilings, states[tau + n], [actions[tau + n]]))
          loss = policy_loss(G, hash_table, num_tilings, action_network, states, actions, tau)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      t += 1
      if tau == T - 1:
        break
    scheduler.step()
    R.append(np.sum(rewards))
  plt.plot(list(range(len(R))), R)
  plt.show()


# ## CartPole

# In[ ]:


### MDP Initialization

env = gym.make('CartPole-v1')

# num_tilings = 8
# n = 10
# epsilon = 0.9
num_action = env.action_space.n
num_dims = len(env.observation_space.low)
scaling = env.observation_space.high - env.observation_space.low
gamma = 0.9
epochs = 100
epsilon = 0.9

num_tilings = [8,10,12]
n_vals = [7,10]
lr_vals = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5]

for tiling in num_tilings:
    for n in n_vals:
        for lr in lr_vals:
            hash_table = hashTable(tiling**(num_dims+num_action))
            action_network = ActionNetwork(tiling)
            optimizer = torch.optim.Adam(action_network.parameters(), lr=lr)
            print(f'Tilings: {tiling}, n: {n}, Learning Rate: {lr}')
            n_step_sarsa(action_network, env, hash_table, tiling, epsilon, n, gamma, epochs, optimizer)


# ## Mountain Car

# In[ ]:


### MDP Initialization

env = gym.make('MountainCar-v0')

# num_tilings = 8
# n = 10
# epsilon = 0.9
num_action = env.action_space.n
num_dims = len(env.observation_space.low)
scaling = env.observation_space.high - env.observation_space.low
gamma = 0.9
epochs = 100
epsilon = 0.9

num_tilings = [8,10,12]
n_vals = [7,10]
lr_vals = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5]

for tiling in num_tilings:
    for n in n_vals:
        for lr in lr_vals:
            hash_table = hashTable(tiling**(num_dims+num_action))
            action_network = ActionNetwork(tiling)
            optimizer = torch.optim.Adam(action_network.parameters(), lr=lr)
            print(f'Tilings: {tiling}, n: {n}, Learning Rate: {lr}')
            n_step_sarsa(action_network, env, hash_table, tiling, epsilon, n, gamma, epochs, optimizer)


# ## Acrobot

# In[ ]:


### MDP Initialization

env = gym.make('Acrobot-v1')

# num_tilings = 8
# n = 10
# epsilon = 0.9
num_action = env.action_space.n
num_dims = len(env.observation_space.low)
scaling = env.observation_space.high - env.observation_space.low
gamma = 0.9
epochs = 100
epsilon = 0.9

num_tilings = [8,10,12]
n_vals = [7,10]
lr_vals = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5]

for tiling in num_tilings:
    for n in n_vals:
        for lr in lr_vals:
            hash_table = hashTable(tiling**(num_dims+num_action))
            action_network = ActionNetwork(tiling)
            optimizer = torch.optim.Adam(action_network.parameters(), lr=lr)
            print(f'Tilings: {tiling}, n: {n}, Learning Rate: {lr}')
            n_step_sarsa(action_network, env, hash_table, tiling, epsilon, n, gamma, epochs, optimizer)

