import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from networks import QNet
import copy


class DQN:
    def __init__(self, gnn_input=40, state_dim=2, ff_hidden=128, epsilon_min=0.05, decay_rate=0.9995, batch_size=512, gamma=0.995, max_memory=10000, tau=0.0025, cuda=False):

        self.device = "cuda" if cuda else "cpu"
        self.tau = tau
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.gnn_input = gnn_input
        self.state_dim = state_dim
        self.replay_count = 0

        self.net = QNet(state_dim, gnn_input, ff_hidden).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimiser = optim.Adam(self.net.parameters(), lr=0.0001)
        self.target_net = copy.deepcopy(self.net)
        self.target_net = self.target_net.to(self.device)

    def act(self, state):
        state, mapping, reverse_mapping = state
        if len(self.memory) < self.batch_size:
            idx = random.randint(0, len(mapping) - 1)
            action = mapping[idx]
            return action, state[idx]

        elif np.random.uniform() < self.epsilon:
            self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)
            idx = random.randint(0, len(state) - 1)
            action = mapping[idx]
            return action, state[idx]

        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_vals = self.net.forward(state)
                idx = int(torch.argmax(q_vals))
            action = mapping[idx]
            return action, state[idx].cpu().numpy()

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, rewards, next_states, dones = self.get_batch()
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_vals = self.net.forward(states)
        targets = self.get_targets(next_states, rewards, dones)

        self.optimiser.zero_grad()
        loss = self.loss_fn(q_vals, targets)
        loss.backward()
        self.optimiser.step()
        self.update_targets()

    def get_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            next_q_vals, _ = torch.max(self.target_net.forward(next_states), dim=1)
        targets = rewards + (1 - dones) * self.gamma * next_q_vals
        return targets

    def get_batch(self):
        batch = random.sample(self.memory, self.batch_size)

        states = [s for s, _, _, _ in batch]
        rewards = [[r] for _, r, _, _ in batch]
        next_states = [ns for _, _, ns, _ in batch]
        dones = [[d] for _, _, _, d in batch]

        return states, rewards, next_states, dones

    def remember(self, state, reward, next_state, done):
        self.memory.append((state, reward, next_state, done))

    def update_targets(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


class GuidedDQN(DQN):
    def __init__(self, gnn_input=40, state_dim=2, ff_hidden=128, epsilon_min=0.05, decay_rate=0.9995, batch_size=512, gamma=0.995, max_memory=10000, tau=0.0025, cuda=False,
                 alpha=0.1):

        super(GuidedDQN, self).__init__(gnn_input, state_dim, ff_hidden, epsilon_min, decay_rate, batch_size, gamma, max_memory, tau, cuda)

        self.alpha = alpha

    def act(self, state):
        state, mapping, reverse_mapping, optimals = state
        if len(self.memory) < self.batch_size:
            idx = random.randint(0, len(mapping) - 1)
            action = mapping[idx]
            return action, state[idx]

        elif np.random.uniform() < self.epsilon:
            self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)
            if np.random.uniform() < self.alpha:
                if optimals:
                    idx = random.sample(optimals, 1)[0]
                    action = mapping[idx]
                else:
                    idx = random.randint(0, len(state) - 1)
                    action = mapping[idx]
                return action, state[idx]

            else:
                idx = random.randint(0, len(mapping) - 1)
                action = mapping[idx]
                return action, state[idx]

        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_vals = self.net.forward(state)
                idx = int(torch.argmax(q_vals))
            action = mapping[idx]
            return action, state[idx].cpu().numpy()
