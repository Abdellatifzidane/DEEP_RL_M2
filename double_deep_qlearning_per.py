"""
Double Deep Q-Network with Prioritized Experience Replay (DDQN-PER)

Prioritized Experience Replay (Schaul et al., 2015) :
- Les transitions avec forte erreur TD sont échantillonnées plus souvent.
- Correction du biais d'échantillonnage via les poids d'importance (IS weights).

Implémentation via SumTree pour un échantillonnage O(log N).

Interface identique à DQNAgent / DoubleDQNAgent existants.
"""

import random
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
# SumTree — structure de données pour PER
# ---------------------------------------------------------------------------

class SumTree:
    """
    Arbre binaire dont chaque nœud interne stocke la somme de ses fils.
    Les feuilles stockent les priorités individuelles.
    Permet un échantillonnage proportionnel en O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity          # nombre de feuilles
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity     # transitions stockées aux feuilles
        self.write = 0                    # pointeur circulaire
        self.n_entries = 0

    # --- Mise à jour interne ---

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # --- Interface publique ---

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1  # index feuille dans l'arbre
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        """Retourne (idx_arbre, priorité, donnée) pour le cumul s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


# ---------------------------------------------------------------------------
# Prioritized Experience Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Buffer PER basé sur SumTree.

    Paramètres :
        capacity      : taille maximale du buffer
        alpha         : degré de prioritisation (0 = uniforme, 1 = full prio)
        beta_start    : valeur initiale de la correction IS
        beta_frames   : nombre de frames pour atteindre beta = 1
        epsilon_prio  : petite constante pour éviter une priorité nulle
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon_prio: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon_prio = epsilon_prio
        self.frame = 1                   # compteur global de transitions ajoutées
        self._max_priority = 1.0         # priorité max observée (pour les nouvelles transitions)

    @property
    def beta(self) -> float:
        """Beta croît linéairement de beta_start à 1."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def add(self, state, action, reward, next_state, next_actions, done):
        # Nouvelle transition → priorité maximale observée
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, next_actions, done))
        self.frame += 1

    def sample(self, batch_size: int):
        """
        Retourne (batch, indices, poids_IS).
        batch   : liste de tuples (s, a, r, s', aa', done)
        indices : indices dans le SumTree (pour update_priorities)
        weights : poids d'importance-sampling normalisés (torch.Tensor 1D)
        """
        batch, indices, priorities = [], [], []
        segment = self.tree.total_priority / batch_size
        beta = self.beta

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            # Garde-fou : données pas encore écrites
            while data is None:
                s = random.uniform(0, self.tree.total_priority)
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(max(priority, self.epsilon_prio))

        # Poids IS :  w_i = (N · P(i))^{-β}  normalisé par max(w)
        total = self.tree.total_priority
        n = len(self.tree)
        weights = np.array([(p / total) * n for p in priorities], dtype=np.float32)
        weights = (weights ** (-beta))
        weights /= weights.max()           # normalisation pour stabilité

        return batch, indices, weights

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Met à jour les priorités après calcul des erreurs TD."""
        for idx, err in zip(indices, td_errors):
            priority = (abs(err) + self.epsilon_prio) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return len(self.tree)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DoubleDeepQLearningWithPrioritizedExperienceReplay:
    """
    Double DQN + Prioritized Experience Replay.

    Double DQN   : online network sélectionne, target network évalue.
    PER          : transitions prioritaires (forte erreur TD) rejouées plus souvent.
    IS weights   : corrigent le biais introduit par l'échantillonnage non-uniforme.
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
        # --- PER spécifiques ---
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
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

        self.q_network     = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
        )
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
    # Conversion action
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
    # Un pas de mise à jour (Double DQN + PER)
    # ------------------------------------------------------------------

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, tree_indices, is_weights = self.replay_buffer.sample(self.batch_size)

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
        weights_t  = torch.tensor(is_weights,         dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q courant
        current_q = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN : sélection online, évaluation target
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

        # Erreurs TD pour mise à jour des priorités
        td_errors = (target_q - current_q).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # Perte pondérée par les poids IS
        elementwise_loss = (current_q - target_q) ** 2
        loss = (weights_t * elementwise_loss).mean()

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
