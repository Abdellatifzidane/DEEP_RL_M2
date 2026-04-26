"""
Double Deep Q-Network with Experience Replay (DDQN-ER)

Différence clé vs DoubleDQNAgent de base :
- Même algorithme Double DQN (online network sélectionne, target network évalue)
- Le ReplayBuffer est explicitement nommé "Experience Replay" pour être cohérent
  avec la taxonomie du cours / rapport
- Interface identique à DQNAgent et DoubleDQNAgent (train, select_action, evaluate)
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Réseau de neurones
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Experience Replay Buffer (uniforme)
# ---------------------------------------------------------------------------

class ExperienceReplayBuffer:
    """Buffer circulaire à échantillonnage uniforme."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_actions, done):
        self.buffer.append((state, action, reward, next_state, next_actions, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DoubleDeepQLearningWithExperienceReplay:
    """
    Double DQN + Experience Replay uniforme.

    Double DQN : le réseau online choisit l'action au prochain état,
    le réseau target l'évalue → réduit la surestimation des Q-valeurs.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 20,
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

        # Réseau online (mis à jour à chaque step)
        self.q_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        # Réseau target (mis à jour périodiquement)
        self.target_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ExperienceReplayBuffer(buffer_capacity)
        self.training_step = 0

    # ------------------------------------------------------------------
    # Helpers état → tenseur
    # ------------------------------------------------------------------

    def _state_to_array(self, state) -> np.ndarray:
        if isinstance(state, tuple):
            flat = []
            for item in state:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            arr = np.array(flat, dtype=np.float32)
        elif hasattr(state, "tolist"):
            arr = np.array(state.tolist(), dtype=np.float32)
        else:
            arr = np.array(state, dtype=np.float32)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.flatten()
        return arr

    def _state_to_tensor(self, state) -> torch.Tensor:
        arr = self._state_to_array(state)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Conversion action tuple ↔ entier
    # Quarto : (type, index) avec type ∈ {0,1}, index ∈ {0..15}
    #   → entier = type * 16 + index   (espace : 0..31)
    # ------------------------------------------------------------------

    @staticmethod
    def _action_to_int(action) -> int:
        if isinstance(action, tuple):
            return action[0] * 16 + action[1]
        return int(action)

    @staticmethod
    def _int_to_action(n: int):
        return (n // 16, n % 16)

    # ------------------------------------------------------------------
    # Sélection d'action (ε-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state, available_actions, training: bool = True):
        if not available_actions:
            return None

        if training and random.random() < self.epsilon:
            return self._action_to_int(random.choice(available_actions))

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)

        best = max(available_actions, key=lambda a: q_values[self._action_to_int(a)].item())
        return self._action_to_int(best)

    # ------------------------------------------------------------------
    # Stockage de transition
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, next_actions, done):
        action_int = self._action_to_int(action)
        next_actions_int = [self._action_to_int(a) for a in next_actions]
        self.replay_buffer.add(
            state=self._state_to_array(state),
            action=action_int,
            reward=reward,
            next_state=self._state_to_array(next_state),
            next_actions=next_actions_int,
            done=done,
        )

    # ------------------------------------------------------------------
    # Un pas de mise à jour (Double DQN)
    # ------------------------------------------------------------------

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, next_actions_batch, dones = zip(*batch)

        def clean(lst):
            return np.stack(
                [s.flatten().astype(np.float32) if isinstance(s, np.ndarray)
                 else np.array(s, dtype=np.float32).flatten()
                 for s in lst], axis=0
            )

        states_t   = torch.tensor(clean(states),      dtype=torch.float32, device=self.device)
        actions_t  = torch.tensor(actions,            dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards_t  = torch.tensor(rewards,            dtype=torch.float32, device=self.device).unsqueeze(1)
        next_st_t  = torch.tensor(clean(next_states), dtype=torch.float32, device=self.device)
        dones_t    = torch.tensor(dones,              dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q courant
        current_q = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN : sélection par online, évaluation par target
            next_q_online  = self.q_network(next_st_t)
            next_q_target  = self.target_network(next_st_t)

            max_next_q_list = []
            for i, valid in enumerate(next_actions_batch):
                if dones_t[i].item() == 1.0 or not valid:
                    max_next_q_list.append(0.0)
                else:
                    best_a = max(valid, key=lambda a: next_q_online[i, a].item())
                    max_next_q_list.append(next_q_target[i, best_a].item())

            max_next_q = torch.tensor(
                max_next_q_list, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return loss.item()

    # ------------------------------------------------------------------
    # Mise à jour réseau target & décroissance epsilon
    # ------------------------------------------------------------------

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Boucle d'entraînement
    # ------------------------------------------------------------------

    def train(self, env, num_episodes: int = 500):
        scores, losses = [], []

        for episode in range(num_episodes):
            state = env.reset()
            ep_losses = []

            while not env.is_terminal():
                available_actions = env.get_available_actions()
                action_int = self.select_action(state, available_actions, training=True)
                action_tuple = self._int_to_action(action_int)

                next_state, reward = env.step(action_tuple)
                done = env.is_terminal()
                next_actions = env.get_available_actions() if not done else []

                self.store_transition(state, action_tuple, reward, next_state, next_actions, done)

                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)

                state = next_state

            if hasattr(env, "get_score"):
                scores.append(env.get_score())
            if ep_losses:
                losses.append(sum(ep_losses) / len(ep_losses))

            self.decay_epsilon()

            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_network()

        return scores, losses

    # ------------------------------------------------------------------
    # Évaluation (greedy, sans exploration)
    # ------------------------------------------------------------------

    def evaluate(self, env, num_episodes: int = 100):
        scores = []
        for _ in range(num_episodes):
            state = env.reset()
            while not env.is_terminal():
                available_actions = env.get_available_actions()
                action_int = self.select_action(state, available_actions, training=False)
                next_state, _ = env.step(self._int_to_action(action_int))
                state = next_state
            if hasattr(env, "get_score"):
                scores.append(env.get_score())
        return sum(scores) / len(scores) if scores else None
