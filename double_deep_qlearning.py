"""
Double Deep Q-Network (Double DQN) Agent
Reduces overestimation of Q-values by using separate target network for action selection.
"""

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


class DoubleDQNAgent:
    """
    Double DQN: uses online network to select action, target network to evaluate.
    Reduces the overestimation problem in standard DQN.
    """
    
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

        # Online network (updated every step)
        self.q_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        
        # Target network (updated periodically)
        self.target_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.training_step = 0

    def _state_to_array(self, state):
        # Handle tuple states (like from Quarto environment)
        # State should be 68-dimensional: 16 cells × 4 attrs + 4 current piece attrs
        if isinstance(state, tuple):
            # Convert tuple elements to a flat list
            flat_list = []
            for item in state:
                if isinstance(item, (list, tuple)):
                    flat_list.extend(item)
                else:
                    flat_list.append(item)
            arr = np.array(flat_list, dtype=np.float32)
        elif hasattr(state, 'tolist'):
            arr = np.array(state.tolist(), dtype=np.float32)
        else:
            arr = np.array(state, dtype=np.float32)
        
        # Ensure 1D array with correct size (68 for Quarto)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.flatten()
            
        return arr

    def _state_to_tensor(self, state):
        state_array = self._state_to_array(state)
        # Ensure proper shape for network input
        if state_array.ndim == 1:
            state_array = state_array.reshape(1, -1)
        return torch.tensor(state_array, dtype=torch.float32, device=self.device)

    def select_action(self, state, available_actions, training=True):
        if not available_actions:
            return None

        if training and random.random() < self.epsilon:
            chosen = random.choice(available_actions)
            # Convert tuple action to integer if needed
            if isinstance(chosen, tuple):
                return chosen[0] * 16 + chosen[1]
            return chosen

        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)

        # Convert tuple actions to integers for Q-value lookup
        def action_to_int(action):
            if isinstance(action, tuple):
                return action[0] * 16 + action[1]
            return action

        best_action = max(available_actions, key=lambda action: q_values[action_to_int(action)].item())
        return action_to_int(best_action)

    def store_transition(self, state, action, reward, next_state, next_actions, done):
        # Convert tuple action to integer if needed
        if isinstance(action, tuple):
            action = action[0] * 16 + action[1]
        
        # Convert next_actions to integers
        next_actions_int = []
        for a in next_actions:
            if isinstance(a, tuple):
                next_actions_int.append(a[0] * 16 + a[1])
            else:
                next_actions_int.append(a)
        
        self.replay_buffer.add(
            state=self._state_to_array(state),
            action=action,
            reward=reward,
            next_state=self._state_to_array(next_state),
            next_actions=next_actions_int,
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

        # Ensure each state is a proper 1D array before stacking
        states_clean = []
        for s in states:
            if isinstance(s, np.ndarray):
                s_flat = s.flatten().astype(np.float32)
            else:
                s_flat = np.array(s, dtype=np.float32).flatten()
            states_clean.append(s_flat)
        
        next_states_clean = []
        for s in next_states:
            if isinstance(s, np.ndarray):
                s_flat = s.flatten().astype(np.float32)
            else:
                s_flat = np.array(s, dtype=np.float32).flatten()
            next_states_clean.append(s_flat)

        # Use np.stack to ensure proper 2D shape (batch_size, state_size)
        states_np = np.stack(states_clean, axis=0)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.stack(next_states_clean, axis=0)
        dones_np = np.array(dones, dtype=np.float32)

        # Create tensors
        states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones_np, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: use online network to select action, target network to evaluate
            next_q_online = self.q_network(next_states)
            next_q_target = self.target_network(next_states)
            
            max_next_q_values = []
            
            for i, valid_actions in enumerate(next_actions_batch):
                if dones[i].item() == 1.0 or not valid_actions:
                    max_next_q_values.append(0.0)
                else:
                    # Select action using online network
                    best_action_idx = max(valid_actions, key=lambda a: next_q_online[i, a].item())
                    # Evaluate with target network
                    target_q = next_q_target[i, best_action_idx].item()
                    max_next_q_values.append(target_q)

            max_next_q = torch.tensor(
                max_next_q_values, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1

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

                # Convert integer action back to tuple for environment
                if isinstance(action, int):
                    action = (action // 16, action % 16)

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
                
                # Convert integer action back to tuple for environment
                if isinstance(action, int):
                    action = (action // 16, action % 16)
                
                next_state, _ = env.step(action)
                state = next_state

            if hasattr(env, "get_score"):
                scores.append(env.get_score())

        if not scores:
            return None

        return sum(scores) / len(scores)