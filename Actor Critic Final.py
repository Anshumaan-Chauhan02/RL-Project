#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt


# ## Actor Critic

# In[ ]:


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
        x = self.fc3(x)
        action_probs = self.softmax(x)
        return action_probs

class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
        x = self.fc3(x)
        state_value = x
        return state_value

def loss_actor(actor, state, action, advantage):
    action_probs = actor(state)
    log_probs = torch.log(action_probs[action])
    actor_loss = -log_probs * advantage
    return actor_loss

def loss_critic(critic, target, state):
    critic_loss = nn.MSELoss()(critic(state), target)
    return critic_loss

def train(actor, critic, actor_optimizer, critic_optimizer, gamma, M, avg_runs, initial_state):
    plots = []
    for run in range(avg_runs):
        list_R = []
        for epoch in range(400):
            steps = 0
            state, _ = env.reset()
            state = torch.Tensor(state)
            R = 0
            log_probs = []
            values = []
            rewards = []

            while True:
                steps+=1
                action_probs = actor(state)
                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = torch.Tensor(next_state)

                next_state_value = critic(next_state)
                td_target = reward + gamma * next_state_value * (1 - terminated)
                td_error = td_target - critic(state)

                advantage = td_error.item()

                actor_optimizer.zero_grad()
                actor_loss = loss_actor(actor, state, action, advantage)
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss = loss_critic(critic, td_target, state)
                critic_loss.backward()
                critic_optimizer.step()

                state = next_state
                R += reward

                if terminated or truncated:
                    break

            list_R.append(R)
        plots.append(list_R)

    plt.plot(list(range(len(plots[0]))), np.mean(plots, axis=0))
    plt.show()


# ### CartPole

# In[ ]:


# MDP Setup
env = gym.make('CartPole-v1')

# Hyperparameter Tuning
lr_vals = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]
# lr_vals = [0.005]
gamma_vals = [0.99]
M_vals = [1]

for lr in lr_vals:
    for gamma in gamma_vals:
        for M in M_vals:
            actor = Actor(4, 2)
            critic = Critic(4)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
            print(f'Learning Rate: {lr}, Gamma: {gamma}, M: {M}')
            print()
            train(actor, critic, actor_optimizer, critic_optimizer, gamma, M, 1, [0, 0, 0, 0])
            print()
            print('-----------------------------------------------------------------')


# ### Mountain Car

# In[ ]:


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
        x = self.fc3(x)
        action_probs = self.softmax(x)
        return action_probs

class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
        x = self.fc3(x)
        state_value = x
        return state_value


# In[ ]:


# MDP Setup
env = gym.make('MountainCar-v0')

# Hyperparameter Tuning
lr_vals = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
# lr_vals = [1e-3]
gamma_vals = [0.99]
M_vals = [1]

for lr in lr_vals:
    for gamma in gamma_vals:
        for M in M_vals:
            actor = Actor(2, 3)
            critic = Critic(2)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
            print(f'Learning Rate: {lr}, Gamma: {gamma}, M: {M}')
            print()
            train(actor, critic, actor_optimizer, critic_optimizer, gamma, M, 1, [0, 0, 0, 0])
            print()
            print('-----------------------------------------------------------------')


# ## Acrobot

# In[ ]:


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
        x = self.fc3(x)
        action_probs = self.softmax(x)
        return action_probs

class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
        x = self.fc3(x)
        state_value = x
        return state_value


# In[ ]:


# MDP Setup
env = gym.make('Acrobot-v1')

# Hyperparameter Tuning
lr_vals = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
# lr_vals = [1e-3]
gamma_vals = [0.99]
M_vals = [1]

for lr in lr_vals:
    for gamma in gamma_vals:
        for M in M_vals:
            actor = Actor(6, 3)
            critic = Critic(6)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
            print(f'Learning Rate: {lr}, Gamma: {gamma}, M: {M}')
            print()
            train(actor, critic, actor_optimizer, critic_optimizer, gamma, M, 1, [0, 0, 0, 0])
            print()
            print('-----------------------------------------------------------------')

