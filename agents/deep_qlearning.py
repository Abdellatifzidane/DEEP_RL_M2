import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_actions, done):
        self.buffer.append((state, action, reward, next_state, next_actions, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=20,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def _state_to_array(self, state):
        return np.array(state, dtype=np.float32)

    def _state_to_tensor(self, state):
        state_array = self._state_to_array(state)
        return torch.tensor(state_array, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state, available_actions, training=True):
        if not available_actions:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(available_actions)

        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)

        best_action = max(available_actions, key=lambda action: q_values[action].item())
        return best_action

    def store_transition(self, state, action, reward, next_state, next_actions, done):
        self.replay_buffer.add(
            state=self._state_to_array(state),
            action=action,
            reward=reward,
            next_state=self._state_to_array(next_state),
            next_actions=list(next_actions),
            done=done,
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        next_actions_batch = []
        dones = []

        for state, action, reward, next_state, next_actions, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions_batch.append(next_actions)
            dones.append(done)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = []

            for i, valid_actions in enumerate(next_actions_batch):
                if dones[i].item() == 1.0 or not valid_actions:
                    max_next_q_values.append(0.0)
                else:
                    best_next_q = max(next_q_values[i, action].item() for action in valid_actions)
                    max_next_q_values.append(best_next_q)

            max_next_q = torch.tensor(
                max_next_q_values, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=500):
        scores = []
        losses = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_losses = []

            while not env.is_terminal():
                available_actions = env.get_available_actions()
                action = self.select_action(state, available_actions, training=True)

                next_state, reward = env.step(action)
                done = env.is_terminal()
                next_actions = env.get_available_actions() if not done else []

                self.store_transition(state, action, reward, next_state, next_actions, done)

                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state

            if hasattr(env, "get_score"):
                scores.append(env.get_score())

            if episode_losses:
                losses.append(sum(episode_losses) / len(episode_losses))

            self.decay_epsilon()

            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_network()

        return scores, losses

    def evaluate(self, env, num_episodes=100):
        scores = []

        for _ in range(num_episodes):
            state = env.reset()

            while not env.is_terminal():
                available_actions = env.get_available_actions()
                action = self.select_action(state, available_actions, training=False)
                next_state, _ = env.step(action)
                state = next_state

            if hasattr(env, "get_score"):
                scores.append(env.get_score())

        if not scores:
            return None

        return sum(scores) / len(scores)