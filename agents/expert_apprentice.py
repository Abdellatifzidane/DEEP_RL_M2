"""
Expert Iteration (ExIt) pour Quarto.

MCTS (expert) génère des données, réseau de neurones (apprentice) apprend
à imiter la distribution de visites. Résultat : un réseau rapide comme PPO
avec la qualité de MCTS.
"""

import math
import random
import multiprocessing as mp
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ──────────────────────────────────────────────────────────────────────────────
# MCTS interne (propre à ExIt, ne touche pas agents/mcts.py)
# ──────────────────────────────────────────────────────────────────────────────

class _MCTSNode:
    __slots__ = ['parent', 'action', 'player', 'children',
                 'wins', 'visits', 'untried_actions']

    def __init__(self, parent=None, action=None, player=None):
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = None

    def uct(self, c):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child_uct(self, c):
        return max(self.children, key=lambda n: n.uct(c))

    def most_visited_child(self):
        return max(self.children, key=lambda n: n.visits)


def _mcts_search(env, num_simulations, c=1.41):
    """Lance num_simulations itérations MCTS et retourne le noeud racine."""
    root = _MCTSNode()
    root.untried_actions = list(env.get_available_actions())
    root.visits = 1

    for _ in range(num_simulations):
        node = root
        sim_env = env.clone()

        # Sélection
        while not node.untried_actions and node.children:
            node = node.best_child_uct(c)
            sim_env.step(node.action)

        # Expansion
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            player = sim_env.current_player
            sim_env.step(action)
            child = _MCTSNode(parent=node, action=action, player=player)
            if not sim_env.is_terminal():
                child.untried_actions = list(sim_env.get_available_actions())
            else:
                child.untried_actions = []
            node.children.append(child)
            node = child

        # Rollout
        while not sim_env.is_terminal():
            actions = sim_env.get_available_actions()
            if not actions:
                break
            sim_env.step(random.choice(actions))

        # Backpropagation
        score = sim_env.get_score()
        winner = sim_env.current_player if score > 0 else None
        while node is not None:
            node.visits += 1
            if winner is not None and node.player == winner:
                node.wins += 1
            node = node.parent

    return root


def _get_visit_probs(root, action_size):
    """Extrait la distribution normalisée des visites → np.array(action_size)."""
    probs = np.zeros(action_size, dtype=np.float32)
    for child in root.children:
        probs[child.action] = child.visits
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# Réseau policy (apprentice)
# ──────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.head(x), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Worker pour la collecte parallèle
# ──────────────────────────────────────────────────────────────────────────────

def _exit_collect_worker(args):
    """Worker : joue num_games avec MCTS pur, retourne [(state, visit_probs)]."""
    (env_class, num_games, num_simulations, c, action_dim, seed) = args

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)

    env = env_class()
    results = []

    for _ in range(num_games):
        env.reset()
        while not env.is_terminal():
            available = env.get_available_actions()
            if not available:
                break

            state = env.encode_state()
            root = _mcts_search(env, num_simulations, c)
            visit_probs = _get_visit_probs(root, action_dim)

            results.append((state.copy(), visit_probs))

            best_action = root.most_visited_child().action
            env.step(best_action)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Agent Expert Iteration
# ──────────────────────────────────────────────────────────────────────────────

class ExpertApprentice:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3,
                 num_simulations=100, c=1.41, buffer_size=50_000,
                 lr_min=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_simulations = num_simulations
        self.c = c

        self.network = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        self.lr_min = lr_min

        self.buffer = deque(maxlen=buffer_size)

    # ── Collecte de données (MCTS expert) ──────────────────────────────────

    def collect_games_parallel(self, env, num_games, num_workers=6):
        """MCTS parallèle : chaque worker joue sa part de games."""
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1
        games_per_worker = [g for g in games_per_worker if g > 0]
        actual_workers = len(games_per_worker)

        args_list = [
            (env.__class__, games_per_worker[i], self.num_simulations,
             self.c, self.action_dim, random.randint(0, 2**31))
            for i in range(actual_workers)
        ]

        ctx = mp.get_context('fork')
        with ctx.Pool(actual_workers) as pool:
            all_results = pool.map(_exit_collect_worker, args_list)

        for results in all_results:
            self.buffer.extend(results)

    def collect_games(self, env, num_games):
        """MCTS joue num_games parties, collecte (state, visit_probs) dans le buffer."""
        for _ in range(num_games):
            env.reset()
            while not env.is_terminal():
                available = env.get_available_actions()
                if not available:
                    break

                state = env.encode_state()
                root = _mcts_search(env, self.num_simulations, self.c)
                visit_probs = _get_visit_probs(root, self.action_dim)

                self.buffer.append((state.copy(), visit_probs))

                best_action = root.most_visited_child().action
                env.step(best_action)

    # ── Entraînement du réseau (apprentice) ────────────────────────────────

    def train_network(self, epochs=5, batch_size=256):
        """Entraîne le réseau avec cross-entropy sur le buffer. Retourne la loss moyenne."""
        if len(self.buffer) < batch_size:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            indices = list(range(len(self.buffer)))
            random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                states = torch.FloatTensor(
                    np.array([self.buffer[i][0] for i in batch_idx])
                )
                targets = torch.FloatTensor(
                    np.array([self.buffer[i][1] for i in batch_idx])
                )

                preds = self.network(states)
                # Cross-entropy : -sum(target * log(pred))
                loss = -(targets * torch.log(preds + 1e-8)).sum(dim=-1).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Décroître le lr (avec plancher)
        if self.optimizer.param_groups[0]['lr'] > self.lr_min:
            self.scheduler.step()

        return total_loss / max(num_batches, 1)

    # ── Sélection d'action (réseau seul, pas MCTS) ────────────────────────

    def select_action(self, state, available_actions):
        """Forward réseau + mask + argmax."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = self.network(state_t)[0].numpy()

        # Masquer les actions invalides
        mask = np.zeros(self.action_dim, dtype=np.float32)
        mask[available_actions] = 1.0
        probs = probs * mask
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[available_actions] = 1.0 / len(available_actions)

        return int(np.argmax(probs))

    # ── Sauvegarde / chargement ────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
