"""
Test DQN et Double DQN sur LineWorld, GridWorld et TicTacToe.
Structure identique au test Quarto (RLTracker, jalons, évaluation greedy).
Architecture réseau : state_size → 256 → ReLU → 256 → ReLU → action_size

Métriques évaluées (policy greedy, pas entraînement) aux jalons :
    1 000 / 10 000 / 100 000 / 200 000 / 300 000 épisodes d'entraînement

TicTacToe  : win_rate / loss_rate / draw_rate  (opponent random intégré dans env)
LineWorld  : win_rate (goal) / loss_rate (trap) / draw_rate (timeout)
GridWorld  : win_rate (goal) / draw_rate (timeout) / loss_rate = 0
"""

import sys
import os
import time
import random
from collections import deque

# Ajouter le répertoire parent au chemin pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environnements.test_env.line_world import LineWorld
from environnements.test_env.grid_world import GridWorld
from environnements.test_env.tic_tac_toe import TicTacToe
from evaluate.tracker import RLTracker

EVAL_MILESTONES = [1_000, 10_000, 100_000, 200_000, 300_000]
NUM_EVAL_EPISODES = 200
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs"
CSV_DIR = "results"

HIDDEN_DIM = 256
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
BUFFER_CAPACITY = 10_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 20

CONFIGS = [
    {
        "name": "config_1",
        "gamma": 0.99,
        "lr": 1e-3,
        "epsilon": EPSILON,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": EPSILON_DECAY,
        "buffer_capacity": BUFFER_CAPACITY,
        "batch_size": BATCH_SIZE,
        "target_update_freq": TARGET_UPDATE_FREQ,
    }
]

#chaque entrée : (class, nom, state_size, action_size)

def _get_env_meta(env_class):
    """Récupère  state_size et action_size depuis l'env."""
    env = env_class()
    env.reset()
    state = env.get_state()

    #state_size
    if isinstance(state, (list, tuple)):
        state_size = len(state)
    elif isinstance(state, np.ndarray):
        state_size = state.size
    else:
        state_size = 1

    if hasattr(env, "action_size"):
        action_size = env.action_size
    else:
        #on parcour qlq états pour trouver le max
        all_actions = set()
        for _ in range(50):
            env.reset()
            for a in env.get_available_actions():
                all_actions.add(a if isinstance(a, int) else a[0] * 16 + a[1])
        action_size = max(all_actions) + 1 if all_actions else 4

    return state_size, action_size

ENVS = [
    (LineWorld,  "LineWorld"),
    (GridWorld,  "GridWorld"),
    (TicTacToe,  "TicTacToe"),
]


#réseau : state_size → 256 → ReLU → 256 → ReLU → action_size

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


#replay buffer

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_actions, done):
        self.buffer.append((state, action, reward, next_state, next_actions, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


#agent base

class _BaseAgent:
    def _arr(self, state) -> np.ndarray:
        """Convertit n'importe quel type d'état en tableau 1D float32."""
        if isinstance(state, np.ndarray):
            return state.flatten().astype(np.float32)
        if isinstance(state, (list, tuple)):
            return np.array(state, dtype=np.float32).flatten()
        # scalaire (LineWorld)
        return np.array([state], dtype=np.float32)

    def _tensor(self, state) -> torch.Tensor:
        return torch.tensor(self._arr(state).reshape(1, -1),
                            dtype=torch.float32, device=self.device)

    def select_action(self, state, available_actions, training: bool = True):
        if not available_actions:
            return None
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        with torch.no_grad():
            q = self.q_network(self._tensor(state)).squeeze(0)
        return max(available_actions, key=lambda a: q[a].item())

    def store_transition(self, state, action, reward, next_state, next_actions, done):
        self.replay_buffer.add(
            state=self._arr(state),
            action=int(action),
            reward=float(reward),
            next_state=self._arr(next_state),
            next_actions=[int(a) for a in next_actions],
            done=done,
        )

    def _build_tensors(self, batch):
        states, actions, rewards, next_states, next_actions_batch, dones = zip(*batch)
        clean = lambda lst: np.stack([
            s.flatten().astype(np.float32) if isinstance(s, np.ndarray)
            else np.array(s, dtype=np.float32).flatten()
            for s in lst
        ])
        return (
            torch.tensor(clean(states), dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(clean(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),
            list(next_actions_batch),
        )

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes: int = 500):
        scores, losses = [], []
        for ep in range(num_episodes):
            state = env.reset()
            ep_losses = []
            while not env.is_terminal():
                avail = env.get_available_actions()
                action = self.select_action(state, avail, training=True)
                next_state, reward = env.step(action)
                done = env.is_terminal()
                next_avail = env.get_available_actions() if not done else []
                self.store_transition(state, action, reward, next_state, next_avail, done)
                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                state = next_state
            if hasattr(env, "get_score"):
                scores.append(env.get_score())
            if ep_losses:
                losses.append(sum(ep_losses) / len(ep_losses))
            self.decay_epsilon()
            if (ep + 1) % self.target_update_freq == 0:
                self.update_target_network()
        return scores, losses

    def evaluate(self, env, num_episodes: int = 100):
        scores = []
        for _ in range(num_episodes):
            state = env.reset()
            while not env.is_terminal():
                avail = env.get_available_actions()
                action = self.select_action(state, avail, training=False)
                state, _ = env.step(action)
            if hasattr(env, "get_score"):
                scores.append(env.get_score())
        return sum(scores) / len(scores) if scores else None


#DQN

class DQNAgent(_BaseAgent):
    def __init__(self, state_size, action_size, hidden_dim=256,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, buffer_capacity=10_000,
                 batch_size=64, target_update_freq=20, device=None):
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
        self.optimizer     = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn       = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        st, ac, rw, nst, dn, nab = self._build_tensors(batch)
        current_q = self.q_network(st).gather(1, ac)

        with torch.no_grad():
            next_q = self.target_network(nst)
            best = [
                0.0 if dn[i].item() == 1.0 or not v
                else max(next_q[i, a].item() for a in v)
                for i, v in enumerate(nab)
            ]
            target_q = rw + (1 - dn) * self.gamma * torch.tensor(
                best, dtype=torch.float32, device=self.device).unsqueeze(1)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()


#DOUBLE DQN

class DoubleDQNAgent(_BaseAgent):
    def __init__(self, state_size, action_size, hidden_dim=256,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, buffer_capacity=10_000,
                 batch_size=64, target_update_freq=20, device=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network      = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer     = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn       = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        st, ac, rw, nst, dn, nab = self._build_tensors(batch)
        current_q = self.q_network(st).gather(1, ac)

        with torch.no_grad():
            next_q_online  = self.q_network(nst)
            next_q_target  = self.target_network(nst)
            best = []
            for i, v in enumerate(nab):
                if dn[i].item() == 1.0 or not v:
                    best.append(0.0)
                else:
                    best_a = max(v, key=lambda a: next_q_online[i, a].item())
                    best.append(next_q_target[i, best_a].item())
            target_q = rw + (1 - dn) * self.gamma * torch.tensor(
                best, dtype=torch.float32, device=self.device).unsqueeze(1)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()


#ÉVALUATION POLICY (greedy)
# win → get_score() == 1
# loss → get_score() == -1 (trap LW, défaite TTT)
# draw → get_score() == 0

def evaluate_policy(agent, env_class, num_eval_episodes: int):
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    total_moves = 0
    total_move_time = 0.0

    for _ in range(num_eval_episodes):
        env = env_class()
        state = env.reset()
        episode_steps = 0

        while not env.is_terminal():
            avail = env.get_available_actions()
            if not avail:
                break

            t0 = time.time()
            action = agent.select_action(state, avail, training=False)
            total_move_time += time.time() - t0
            total_moves += 1

            if action is None:
                break

            state, reward = env.step(action)
            episode_steps += 1

        score = env.get_score() if hasattr(env, "get_score") else 0
        if score == 1:
            wins   += 1
        elif score == -1:
            losses += 1
        else:
            draws  += 1

        total_steps += episode_steps

    avg_score = wins   / num_eval_episodes
    loss_rate = losses / num_eval_episodes
    draw_rate = draws  / num_eval_episodes
    avg_steps = total_steps / num_eval_episodes
    avg_move_time = total_move_time / total_moves if total_moves > 0 else 0.0

    return avg_score, loss_rate, draw_rate, avg_steps, avg_move_time


#entraînement incremental + évaluation à chaque jalon

def run(agent_class, agent_label: str, config: dict,
        env_class, env_name: str, state_size: int, action_size: int):

    config_name  = config["name"]
    agent_kwargs = {k: v for k, v in config.items() if k != "name"}
    agent_kwargs.update({"state_size": state_size,
                         "action_size": action_size,
                         "hidden_dim": HIDDEN_DIM})

    agent = agent_class(**agent_kwargs)
    tracker = RLTracker(
        agent_name=f"{agent_label}_{config_name}",
        env_name=env_name,
        config={
            **config,
            "agent": agent_label,
            "hidden_dim": HIDDEN_DIM,
            "eval_milestones": EVAL_MILESTONES,
            "num_eval_episodes": NUM_EVAL_EPISODES,
        },
        log_dir=LOG_DIR,
        csv_dir=CSV_DIR,
    )

    print(f"\n{'='*65}")
    print(f"{agent_label} | {env_name} | {config_name}")
    print(f"Réseau : {state_size} → 256 → ReLU → 256 → ReLU → {action_size}")
    print(f"{'='*65}")

    env= env_class()
    cumulative= 0

    try:
        for milestone in EVAL_MILESTONES:
            episodes_needed = milestone - cumulative
            if episodes_needed <= 0:
                continue

            print(
                f"\n[Train] {cumulative:>7,} → {milestone:>7,} "
                f"({episodes_needed:,} épisodes)...",
                end="", flush=True,
            )
            t0 = time.time()
            agent.train(env, num_episodes=episodes_needed)
            print(f"✓ {time.time() - t0:.1f}s")

            # --- Évaluation greedy ---
            print(
                f"[Eval] {NUM_EVAL_EPISODES} épisodes...",
                end="", flush=True,
            )
            start_total_time = time.time()

            avg_score, loss_rate, draw_rate, avg_steps, avg_move_time = evaluate_policy(
                agent=agent,
                env_class=env_class,
                num_eval_episodes=NUM_EVAL_EPISODES,
            )

            total_time = time.time() - start_total_time

            print(f"[DEBUG] LOGGED STEP = {milestone}")
            tracker.log_evaluation(
                num_episodes_trained=milestone,
                avg_score=avg_score,
                avg_steps=avg_steps,
                loss_rate=loss_rate,
                draw_rate=draw_rate,
                avg_move_time_sec=avg_move_time,
                total_eval_time_sec=total_time,
            )
            tracker.writer.flush()

            print(
                f"[EVAL] Episode {milestone} | "
                f"win_rate={avg_score:.4f} | "
                f"loss_rate={loss_rate:.4f} | "
                f"draw_rate={draw_rate:.4f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"avg_move_time={avg_move_time:.6f}s | "
                f"total_time={total_time:.2f}s"
            )

            cumulative = milestone

    finally:
        tracker.finish()

def main():
    print("=" * 65)
    print("DQN & Double DQN — LineWorld / GridWorld / TicTacToe")
    print(f"Réseau : état → 256 → ReLU → 256 → ReLU → actions")
    print(f"Jalons : {EVAL_MILESTONES}")
    print(f"Éval/jalon : {NUM_EVAL_EPISODES} épisodes")
    print("=" * 65)

    agents = [
        (DQNAgent,"DQN"),
        (DoubleDQNAgent, "DoubleDQN"),
    ]

    for env_class, env_name in ENVS:
        state_size, action_size = _get_env_meta(env_class)
        print(f"\n{'▶  ' + env_name + f'  (state={state_size}, actions={action_size})':^65}")

        for agent_class, agent_label in agents:
            for config in CONFIGS:
                run(
                    agent_class=agent_class,
                    agent_label=agent_label,
                    config=config,
                    env_class=env_class,
                    env_name=env_name,
                    state_size=state_size,
                    action_size=action_size,
                )

    print("\n✓ Terminé.")

if __name__ == "__main__":
    main()
