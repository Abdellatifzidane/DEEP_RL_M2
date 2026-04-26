"""
AlphaZero pour Quarto.

MCTS guidé par réseau policy+value (PUCT, pas de rollouts).
Le réseau et le MCTS s'améliorent mutuellement.
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
# Réseau policy + value (deux têtes, même archi que ExIt/PPO)
# ──────────────────────────────────────────────────────────────────────────────

class PolicyValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value


# ──────────────────────────────────────────────────────────────────────────────
# MCTS avec PUCT (guidé par le réseau, pas de rollout)
# ──────────────────────────────────────────────────────────────────────────────

class _AZNode:
    __slots__ = ['parent', 'action', 'player', 'children',
                 'prior', 'visit_count', 'value_sum']

    def __init__(self, parent=None, action=None, player=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.player = player       # joueur qui a joué l'action menant ici
        self.children = []
        self.prior = prior          # P(s,a) donné par le réseau
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, c_puct, parent_visits):
        """PUCT = Q + c × P × √N_parent / (1 + N)"""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct):
        return max(self.children, key=lambda ch: ch.puct_score(c_puct, self.visit_count))


def _expand_node(node, env, network):
    """Expand un noeud : réseau → priors pour chaque action légale. Retourne v(s)."""
    state = env.encode_state()
    available = env.get_available_actions()
    if not available:
        return 0.0

    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        policy, value = network(state_t)
        policy = policy[0].numpy()
        v = value[0].item()

    # Mask + renormalise
    mask = np.zeros(len(policy), dtype=np.float32)
    mask[available] = 1.0
    policy = policy * mask
    total = policy.sum()
    if total > 0:
        policy /= total
    else:
        policy[available] = 1.0 / len(available)

    for a in available:
        child = _AZNode(parent=node, action=a, player=env.current_player,
                        prior=policy[a])
        node.children.append(child)

    return v


def _az_search(env, network, num_simulations, c_puct=1.5,
               dirichlet_alpha=0.3, dirichlet_eps=0.25):
    """MCTS AlphaZero : PUCT + réseau (pas de rollout)."""
    root = _AZNode()

    # Expansion de la racine
    v = _expand_node(root, env, network)
    if not root.children:
        return root

    # Bruit de Dirichlet à la racine (self-play uniquement, pas en éval)
    if dirichlet_eps > 0 and dirichlet_alpha > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, child in enumerate(root.children):
            child.prior = (1 - dirichlet_eps) * child.prior + dirichlet_eps * noise[i]

    root.visit_count = 1

    for _ in range(num_simulations):
        node = root
        sim_env = env.clone()

        # 1. Sélection : descendre avec PUCT
        while node.children:
            node = node.best_child(c_puct)
            sim_env.step(node.action)

        # 2. Évaluation de la feuille
        if sim_env.is_terminal():
            # Noeud terminal → résultat réel
            score = sim_env.get_score()
            if score > 0:
                leaf_value = 1.0
                leaf_player = sim_env.current_player   # gagnant
            else:
                leaf_value = 0.0
                leaf_player = None                     # match nul
        else:
            # 3. Expansion : réseau donne (priors, valeur)
            leaf_value = _expand_node(node, sim_env, network)
            leaf_player = sim_env.current_player       # v est de son point de vue

        # 4. Backpropagation
        while node is not None:
            node.visit_count += 1
            if leaf_player is not None:
                if node.player == leaf_player:
                    node.value_sum += leaf_value
                else:
                    node.value_sum -= leaf_value
            node = node.parent

    return root


def _get_visit_probs(root, action_size, temperature=1.0):
    """Distribution de visites avec température."""
    visits = np.zeros(action_size, dtype=np.float32)
    for child in root.children:
        visits[child.action] = child.visit_count

    if temperature < 0.01:
        # Greedy
        probs = np.zeros_like(visits)
        if visits.max() > 0:
            probs[np.argmax(visits)] = 1.0
        return probs

    # Normaliser les visites avant d'appliquer la température (évite overflow)
    max_v = visits.max()
    if max_v == 0:
        return visits
    log_visits = np.log(visits + 1e-8) / temperature
    log_visits -= log_visits.max()          # stabilité numérique
    visits_temp = np.exp(log_visits)
    visits_temp[visits == 0] = 0.0          # garder les zéros
    total = visits_temp.sum()
    if total > 0:
        return visits_temp / total
    return visits


# ──────────────────────────────────────────────────────────────────────────────
# Worker pour le self-play parallèle
# ──────────────────────────────────────────────────────────────────────────────

def _self_play_worker(args):
    """Fonction worker : joue num_games en self-play, retourne [(state, pi, z)]."""
    (env_class, state_dict, state_dim, action_dim, hidden_dim,
     num_games, num_sims, c_puct, dir_alpha, dir_eps,
     temp_threshold, seed) = args

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)

    network = PolicyValueNetwork(state_dim, action_dim, hidden_dim)
    network.load_state_dict(state_dict)
    network.eval()

    env = env_class()
    results = []

    for _ in range(num_games):
        env.reset()
        game_data = []
        move_count = 0

        while not env.is_terminal():
            available = env.get_available_actions()
            if not available:
                break

            state = env.encode_state()
            root = _az_search(env, network, num_sims, c_puct, dir_alpha, dir_eps)

            temp = 1.0 if move_count < temp_threshold else 0.01
            pi = _get_visit_probs(root, action_dim, temperature=temp)

            game_data.append((state.copy(), pi.copy(), env.current_player))

            if temp > 0.01:
                action = np.random.choice(action_dim, p=pi)
            else:
                action = int(np.argmax(pi))

            env.step(action)
            move_count += 1

        score = env.get_score()
        winner = env.current_player if score > 0 else None

        for s, p, player in game_data:
            if winner is None:
                z = 0.0
            elif player == winner:
                z = 1.0
            else:
                z = -1.0
            results.append((s, p, z))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Agent AlphaZero
# ──────────────────────────────────────────────────────────────────────────────

class AlphaZeroAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3,
                 num_simulations=100, c_puct=1.5, buffer_size=50_000,
                 lr_min=1e-5, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 temp_threshold=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temp_threshold = temp_threshold   # après N coups → greedy

        self.hidden_dim = hidden_dim

        self.network = PolicyValueNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr,
                                    weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        self.lr_min = lr_min

        self.buffer = deque(maxlen=buffer_size)

    # ── Self-play (MCTS + réseau) ─────────────────────────────────────────

    def collect_games_parallel(self, env, num_games, num_workers=6):
        """Self-play parallèle : chaque worker joue sa part de games."""
        self.network.eval()
        state_dict = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}

        # Répartir les games entre workers
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1
        games_per_worker = [g for g in games_per_worker if g > 0]
        actual_workers = len(games_per_worker)

        args_list = [
            (env.__class__, state_dict,
             self.state_dim, self.action_dim, self.hidden_dim,
             games_per_worker[i], self.num_simulations, self.c_puct,
             self.dirichlet_alpha, self.dirichlet_eps,
             self.temp_threshold, random.randint(0, 2**31))
            for i in range(actual_workers)
        ]

        ctx = mp.get_context('fork')
        with ctx.Pool(actual_workers) as pool:
            all_results = pool.map(_self_play_worker, args_list)

        for results in all_results:
            self.buffer.extend(results)

        self.network.train()

    def collect_games(self, env, num_games):
        """Self-play : MCTS guidé par le réseau, collecte (state, π, z)."""
        self.network.eval()

        for _ in range(num_games):
            env.reset()
            game_data = []      # (state, pi_mcts, current_player)
            move_count = 0

            while not env.is_terminal():
                available = env.get_available_actions()
                if not available:
                    break

                state = env.encode_state()
                root = _az_search(
                    env, self.network, self.num_simulations,
                    self.c_puct, self.dirichlet_alpha, self.dirichlet_eps,
                )

                # Température : explorer au début, exploiter ensuite
                temp = 1.0 if move_count < self.temp_threshold else 0.01
                pi = _get_visit_probs(root, self.action_dim, temperature=temp)

                game_data.append((state.copy(), pi.copy(), env.current_player))

                # Échantillonner selon π
                if temp > 0.01:
                    action = np.random.choice(self.action_dim, p=pi)
                else:
                    action = int(np.argmax(pi))

                env.step(action)
                move_count += 1

            # Résultat de la partie → z
            score = env.get_score()
            winner = env.current_player if score > 0 else None

            for state, pi, player in game_data:
                if winner is None:
                    z = 0.0
                elif player == winner:
                    z = 1.0
                else:
                    z = -1.0
                self.buffer.append((state, pi, z))

        self.network.train()

    # ── Entraînement ──────────────────────────────────────────────────────

    def train_network(self, epochs=5, batch_size=256):
        """Entraîne policy+value. Retourne (loss, policy_loss, value_loss)."""
        if len(self.buffer) < batch_size:
            return 0.0, 0.0, 0.0

        total_loss = 0.0
        total_p = 0.0
        total_v = 0.0
        n_batches = 0

        for _ in range(epochs):
            indices = list(range(len(self.buffer)))
            random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                states = torch.FloatTensor(
                    np.array([self.buffer[i][0] for i in batch_idx])
                )
                target_pi = torch.FloatTensor(
                    np.array([self.buffer[i][1] for i in batch_idx])
                )
                target_z = torch.FloatTensor(
                    np.array([self.buffer[i][2] for i in batch_idx])
                ).unsqueeze(1)

                pred_pi, pred_v = self.network(states)

                # Policy : cross-entropy  −Σ π_mcts · log(π_réseau)
                p_loss = -(target_pi * torch.log(pred_pi + 1e-8)).sum(dim=-1).mean()
                # Value : MSE  (v − z)²
                v_loss = F.mse_loss(pred_v, target_z)

                loss = p_loss + v_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_p += p_loss.item()
                total_v += v_loss.item()
                n_batches += 1

        # Décroissance lr (avec plancher)
        if self.optimizer.param_groups[0]['lr'] > self.lr_min:
            self.scheduler.step()

        n = max(n_batches, 1)
        return total_loss / n, total_p / n, total_v / n

    # ── Sélection d'action (MCTS + réseau, le vrai AlphaZero) ───────────

    def select_action(self, env, num_simulations=None):
        """MCTS + réseau → argmax des visites. Pas de Dirichlet en éval."""
        self.network.eval()
        n_sims = num_simulations or self.num_simulations

        root = _az_search(
            env, self.network, n_sims, self.c_puct,
            dirichlet_alpha=0.0, dirichlet_eps=0.0,
        )

        if not root.children:
            available = env.get_available_actions()
            return random.choice(available) if available else 0

        best = max(root.children, key=lambda ch: ch.visit_count)
        return best.action

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.network.load_state_dict(ckpt['network'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
