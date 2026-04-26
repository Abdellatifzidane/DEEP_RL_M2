"""
Tournoi Head-to-Head Quarto.

Round-robin entre 5 agents (AlphaZero, ExIt, PPO, MCTS, Random).
Chaque paire joue 200 games (100 en P0, 100 en P1).
"""

import os
import sys
import time
import random
import csv
from itertools import combinations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from environnements.quarto.quatro import Quatro, ACTION_SIZE, STATE_SIZE
from agents.alpha_zero import AlphaZeroAgent
from agents.expert_apprentice import ExpertApprentice
from agents.ppo_a2c import PPO_A2C
from agents.mcts import MCTSAgent


# ── Configuration ─────────────────────────────────────────────────────────────

GAMES_PER_SIDE = 100          # 100 en P0 + 100 en P1 = 200 par matchup
MCTS_SIMULATIONS = 100        # sims pour l'agent MCTS pur
AZ_EVAL_SIMULATIONS = 200     # sims pour AlphaZero en éval

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

AGENT_CONFIGS = {
    "AlphaZero": {
        "type": "alphazero",
        "model": os.path.join(MODEL_DIR, "az_quarto_100000g.pt"),
    },
    "ExIt": {
        "type": "exit",
        "model": os.path.join(MODEL_DIR, "exit_quarto_100000g.pt"),
    },
    "PPO": {
        "type": "ppo",
        "model": os.path.join(MODEL_DIR, "ppo_quarto_100000ep.pt"),
    },
    "MCTS": {
        "type": "mcts",
        "model": None,
    },
    "Random": {
        "type": "random",
        "model": None,
    },
}


# ── Chargement des agents ─────────────────────────────────────────────────────

def load_agents():
    agents = {}

    # AlphaZero
    az = AlphaZeroAgent(
        state_dim=STATE_SIZE, action_dim=ACTION_SIZE,
        hidden_dim=256, num_simulations=AZ_EVAL_SIMULATIONS,
    )
    az.load(AGENT_CONFIGS["AlphaZero"]["model"])
    az.network.eval()
    agents["AlphaZero"] = az

    # ExIt
    exit_agent = ExpertApprentice(
        state_dim=STATE_SIZE, action_dim=ACTION_SIZE, hidden_dim=256,
    )
    exit_agent.load(AGENT_CONFIGS["ExIt"]["model"])
    exit_agent.network.eval()
    agents["ExIt"] = exit_agent

    # PPO
    ppo = PPO_A2C(
        state_dim=STATE_SIZE, action_dim=ACTION_SIZE, hidden_dim=256,
    )
    ppo.load(AGENT_CONFIGS["PPO"]["model"])
    ppo.policy.eval()
    agents["PPO"] = ppo

    # MCTS
    agents["MCTS"] = MCTSAgent(num_simulations=MCTS_SIMULATIONS)

    # Random (pas d'objet, on gère dans get_action)
    agents["Random"] = None

    return agents


# ── Sélection d'action unifiée ────────────────────────────────────────────────

def get_action(agent_name, agent, env, state, available_actions):
    atype = AGENT_CONFIGS[agent_name]["type"]

    if atype == "alphazero":
        return agent.select_action(env)

    elif atype == "mcts":
        return agent.choose_action(env)

    elif atype == "exit":
        return agent.select_action(state, available_actions)

    elif atype == "ppo":
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = agent.policy(st)
            p = probs[0].numpy()
            # Masquer les actions invalides
            mask = np.zeros_like(p)
            mask[available_actions] = 1.0
            p = p * mask
            total = p.sum()
            if total > 0:
                p /= total
            else:
                p[available_actions] = 1.0 / len(available_actions)
            return int(np.argmax(p))

    else:  # random
        return random.choice(available_actions)


# ── Jouer une partie ──────────────────────────────────────────────────────────

def play_game(name_a, agent_a, name_b, agent_b, env):
    """Joue une partie. agent_a = P0, agent_b = P1.
    Retourne : 'a' si A gagne, 'b' si B gagne, 'draw' sinon."""
    env.reset()

    while not env.is_terminal():
        available = env.get_available_actions()
        if not available:
            break

        state = env.encode_state()
        current = env.current_player  # 0 ou 1

        if current == 0:
            action = get_action(name_a, agent_a, env, state, available)
        else:
            action = get_action(name_b, agent_b, env, state, available)

        env.step(action)

    score = env.get_score()
    if score > 0:
        # current_player au moment du terminal = le gagnant
        winner_player = env.current_player
        return 'a' if winner_player == 0 else 'b'
    return 'draw'


# ── Matchup entre deux agents ─────────────────────────────────────────────────

def run_matchup(name_a, agent_a, name_b, agent_b, games_per_side):
    """Joue games_per_side en P0/P1 pour chaque agent.
    Retourne (wins_a, wins_b, draws)."""
    wins_a = 0
    wins_b = 0
    draws = 0
    env = Quatro()

    # A en P0, B en P1
    for _ in range(games_per_side):
        result = play_game(name_a, agent_a, name_b, agent_b, env)
        if result == 'a':
            wins_a += 1
        elif result == 'b':
            wins_b += 1
        else:
            draws += 1

    # B en P0, A en P1 (on inverse les rôles)
    for _ in range(games_per_side):
        result = play_game(name_b, agent_b, name_a, agent_a, env)
        if result == 'a':
            wins_b += 1  # B est en P0 ici
        elif result == 'b':
            wins_a += 1  # A est en P1 ici
        else:
            draws += 1

    return wins_a, wins_b, draws


# ── Tournoi round-robin ──────────────────────────────────────────────────────

def run_tournament():
    print("=" * 70)
    print("  TOURNOI HEAD-TO-HEAD QUARTO")
    print("  5 agents, round-robin, {} games par matchup".format(GAMES_PER_SIDE * 2))
    print("=" * 70)

    print("\nChargement des agents...")
    agents = load_agents()
    agent_names = list(AGENT_CONFIGS.keys())
    print("Agents charges :", ", ".join(agent_names))

    # Stocker les résultats
    matchup_results = []
    total_wins = {name: 0 for name in agent_names}

    pairs = list(combinations(agent_names, 2))
    print(f"\n{len(pairs)} matchups a jouer\n")

    for i, (name_a, name_b) in enumerate(pairs):
        t0 = time.time()
        print(f"[{i+1}/{len(pairs)}] {name_a} vs {name_b} ...", end=" ", flush=True)

        wins_a, wins_b, draws = run_matchup(
            name_a, agents[name_a],
            name_b, agents[name_b],
            GAMES_PER_SIDE,
        )

        elapsed = time.time() - t0
        total_games = GAMES_PER_SIDE * 2
        print(f"{name_a} {wins_a}W - {wins_b}W {name_b}  ({draws}D)  [{elapsed:.1f}s]")

        matchup_results.append({
            "agent_a": name_a,
            "agent_b": name_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "total": total_games,
        })

        total_wins[name_a] += wins_a
        total_wins[name_b] += wins_b

    # ── Affichage final ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTATS PAR MATCHUP")
    print("=" * 70)
    print(f"{'Matchup':<30} {'W_A':>5} {'W_B':>5} {'Draw':>5} {'Total':>6}")
    print("-" * 70)
    for m in matchup_results:
        label = f"{m['agent_a']} vs {m['agent_b']}"
        print(f"{label:<30} {m['wins_a']:>5} {m['wins_b']:>5} {m['draws']:>5} {m['total']:>6}")

    # Classement
    ranking = sorted(total_wins.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 70)
    print("  CLASSEMENT FINAL (total victoires)")
    print("=" * 70)
    print(f"{'Rang':<6} {'Agent':<15} {'Victoires':>10}")
    print("-" * 35)
    for rank, (name, wins) in enumerate(ranking, 1):
        print(f"{rank:<6} {name:<15} {wins:>10}")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "quarto_tournament_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent_a", "agent_b", "wins_a", "wins_b", "draws", "total",
        ])
        writer.writeheader()
        writer.writerows(matchup_results)

    print(f"\nResultats sauvegardes dans {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_tournament()
