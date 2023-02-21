import gym
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from copy import deepcopy
from network_functionality import Passing
from replay_buffer_class import ReplayBuffer

filehandler = open("best_network.obj", 'rb') 
actor_network = pickle.load(filehandler)
env = gym.make("BipedalWalker-v3")

passing = Passing()
episodes = 101
best_reward = float("-inf")

FC1_DIMS = 1024
FC2_DIMS = 512
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
DEVICE = torch.device("cpu")

best_reward = -130
average_reward = 0
episode_number = []
average_reward_number = []

class critic_network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_space = env.action_space.shape[0]    
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.input_shape = self.num_states + self.num_actions

        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent:
    def __init__(self):
        self.memory = ReplayBuffer()
        self.tau = 1e-2
        self.loss_recorded = 0

        self.actor = deepcopy(actor_network)
        self.actor_target = deepcopy(actor_network)
        self.critic = critic_network()
        self.critic_target = critic_network()

    def choose_action(self, state):
        passing.network = self.actor.network
        action = passing.forward(state)
        action = np.clip(action, -1, 1)

        return action

    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)

        values = self.critic(states, actions)

        # use target actor
        actions_ = []
        for i in range(BATCH_SIZE):
            passing.network = self.actor_target.network
            actions_.append(passing.forward([states_[i][4:8]]))
        actions_ = torch.tensor(actions_)
        
        values_ = self.critic_target(states_, actions_)
        values_[dones] = 0.0

        target = rewards + GAMMA * values_ 
        td = target - values       

        self.critic.train()
        critic_loss = (td**2).mean()
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        for i in range(BATCH_SIZE):
            retrospective_action = np.array(self.choose_action([states[i][4:8]])).astype(np.float32)
            retrospective_action = torch.tensor(retrospective_action)

            retrospective_value = self.critic(states[i].unsqueeze(0), retrospective_action.unsqueeze(0))
            actor_loss = -retrospective_value

            error = retrospective_action * actor_loss

            passing.network = self.actor.network
            passing.backward(error)
            self.actor.network = passing.network

            
        
        """
        error = actor_loss.item()
        actor_loss = np.array([error, error, error, error])
        passing.network = self.actor.network

        passing.backward(actor_loss)
        self.actor.network = passing.network
        """

    
        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        critic_params = self.critic.named_parameters()
        critic_target_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        critic_target_dict = dict(critic_target_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*critic_target_dict[name].clone()

        self.critic_target.load_state_dict(critic_state_dict)

        for layer_idx in range(len(self.actor.network)):
            for node_idx in range(len(self.actor.network[layer_idx])):
                for connection_idx in range(len(self.actor.network[layer_idx][node_idx].connections)):
                    self.actor.network[layer_idx][node_idx].connections[connection_idx].weight = tau * self.actor.network[layer_idx][node_idx].connections[connection_idx].weight + \
                        (1-tau) * self.actor_target.network[layer_idx][node_idx].connections[connection_idx].weight
                    
        self.actor_target = self.actor


agent = Agent()

for episode in range(1, episodes):
    state = env.reset()
    score = 0
    step = 0

    while True:
        env.render()
        step += 1

        action = np.array(agent.choose_action([state[4:8]])).astype(np.float32)
        #action = env.action_space.sample()
        #action = agent.noise.get_action(action, step)


        #if np.isnan(action).any():
            #action = agent.choose_action([state[4:8]])

        state_, reward, done, info = env.step(action)
        agent.memory.add(state, action, reward, state_, done)
        agent.learn()
        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {}".format(episode, average_reward/episode, best_reward, score))
            break
        
    episode_number.append(episode)
    average_reward_number.append(average_reward/episode)

    env.close()

filehandler = open("best_network2.obj", 'wb') 
pickle.dump(agent.actor, filehandler)
filehandler = open("best_network2.obj", 'rb') 
best_run = pickle.load(filehandler)

plt.plot(episode_number, average_reward_number, color= "black", marker= "x")
#slope, intercept = np.polyfit(episode_number, average_reward_number, 1)
#plt.plot(episode_number, slope * np.array(episode_number) + intercept)

plt.show()
