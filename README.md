<h2>
<p align='center'>
Reinforcement Learning (RL) Project
</p>
</h2>

<h4 align='center'> Project Description </h4>
We employ the Actor-Critic, Reinforce with Baseline, and Episodic n-step SARSA algorithms to acquire an optimal policy for distinct Markov Decision Processes (MDPs), specifically, MountainCar-v0, Acrobot-v0, and CarPole-v1 from the OpenAI Gym library. Systematic experimentation has been conducted on both hyperparameters and model architecture, leading in the presentation of results for the most effective configuration.

### Technical Skills
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI Gym](https://img.shields.io/badge/OpenAI_Gym-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
<br>


### Dependencies
##### OpenAI Gym
      !pip install gym
##### PyTorch (Check CPU/GPU Compatibility)
      https://pytorch.org/get-started/locally/
##### NumPy
      !pip install numpy
##### Matplotlib
      !pip install matplotlib

### File Contents
* Actor Critic Final.py
  - Contains the implementation of the Actor-Critic algorithm, a reinforcement learning technique combining policy (Actor) and value function (Critic) approximation to enhance learning efficiency.
* REINFORCE with Baseline Final.py:
  - Encompasses the implementation of the REINFORCE algorithm with Baseline, a policy gradient method incorporating a baseline to reduce variance in gradient estimates.
* Semi-Gradient-SARSA Final.py
  - Houses the implementation of the Semi-Gradient-SARSA algorithm, a temporal difference learning method applied in reinforcement learning scenarios for updating Q-values and optimizing policy.
