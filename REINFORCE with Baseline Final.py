#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

# Importing necessary libraries
import warnings
warnings.simplefilter('ignore')


# ## REINFORCE with Baseline

# In[ ]:


# Loss Function
def get_error_vals(actor, critic, states, actions, log_probs, rewards, gamma, t):
    estimated_value = critic(states[t])
    G = torch.sum(torch.FloatTensor([(gamma**(k-t-1))*r for k, r in enumerate(rewards[t:])]))
    td_error = G - estimated_value
    return td_error, G, estimated_value

# Training Loop
def train(actor, critic, actor_optimizer, critic_optimizer, gamma):
    list_R = []
    for epoch in range(100):
        state, _ = env.reset()
        state = torch.Tensor(state)
        log_probs = []
        states = [state]
        values = []
        rewards = []
        actions = []

        while True:
            action_probs = actor(state)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.Tensor(next_state)

            log_probs.append(torch.log(action_probs[action]))
            values.append(critic(state))
            rewards.append(reward)
            actions.append(action)

            state = next_state
            states.append(state)

            if terminated or truncated:
                break

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        for t in range(len(actions)):
            td_error, G, estimated_value = get_error_vals(actor, critic, torch.stack(states),
                                                        torch.tensor(actions),
                                                        torch.stack(log_probs),
                                                        rewards,
                                                        gamma,
                                                        t)

            actor_loss = -1 * (gamma ** t) * (log_probs[t] * td_error)
            critic_loss = 0.5 * (td_error ** 2)

            actor_loss.backward(retain_graph=True)
            critic_loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()

        list_R.append(np.sum(rewards))

    plt.plot(range(len(list_R)), list_R)
    plt.show()


# In[ ]:


# Actor Network
class Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        action_probs = nn.functional.softmax(x, dim=-1)
        return action_probs

# Critic Network
class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ## CartPole

# In[ ]:


# MDP Setup
env = gym.make('CartPole-v1')

# Hyperparameter Tuning
lr_vals = [1e-4, 1e-3, 1e-2, 1e-1]
gamma = 0.99
for lr in lr_vals:
    actor = Actor(4,2)
    critic = Critic(4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    print(f'Learning Rate: {lr}, Gamma: {gamma}')
    print()
    train(actor, critic, actor_optimizer, critic_optimizer, gamma=gamma)
    print()
    print('-----------------------------------------------------------------')


# ## Mountain Car

# In[ ]:


# MDP Setup
env = gym.make('MountainCar-v0')

# Hyperparameter Tuning
lr_vals = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1]
gamma = 0.99
for lr in lr_vals:
    actor = Actor(2,2)
    critic = Critic(2)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    print(f'Learning Rate: {lr}, Gamma: {gamma}')
    print()
    train(actor, critic, actor_optimizer, critic_optimizer, gamma=gamma)
    print()
    print('-----------------------------------------------------------------')


# ## Acrobot

# In[ ]:


# MDP Setup
env = gym.make('Acrobot-v1')

# Hyperparameter Tuning
lr_vals = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1]
gamma = 0.99
for lr in lr_vals:
    actor = Actor(6,3)
    critic = Critic(6)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    print(f'Learning Rate: {lr}, Gamma: {gamma}')
    print()
    train(actor, critic, actor_optimizer, critic_optimizer, gamma=gamma)
    print()
    print('-----------------------------------------------------------------')


# In[ ]:




