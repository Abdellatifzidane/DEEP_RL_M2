"""
Test Expert Iteration (ExIt) sur Quarto.

Boucle : collect batch (MCTS) → train réseau → évaluer aux paliers.
"""

import os
import sys
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from environnements.quarto.quatro import Quatro, ACTION_SIZE, STATE_SIZE
from agents.expert_apprentice import ExpertApprentice
from evaluate.tracker import RLTracker


TEST_CONFIGS = [
    {
        "name": "exit_100sims_v2",
        "state_dim": STATE_SIZE,
        "action_dim": ACTION_SIZE,
        "hidden_dim": 256,
        "lr": 1e-3,
        "num_simulations": 100,
        "games_per_batch": 500,
        "total_games": 100_000,
        "train_epochs": 10,
        "train_batch_size": 256,
        "eval_games": 100,
        "eval_every": 1_000,
        "buffer_size": 100_000,
        "num_workers": 6,
    },
]


def evaluate_vs_random(agent, env, num_games=100):
    """Évalue le réseau (pas MCTS) contre un joueur random.
    Retourne (score_moyen, longueur_moyenne)."""
    total_score = 0
    total_steps = 0

    for game in range(num_games):
        state = env.reset()
        agent_plays_first = game % 2 == 0
        steps = 0

        while not env.is_terminal():
            available_actions = env.get_available_actions()
            if not available_actions:
                break

            if (env.current_player == 0 and agent_plays_first) or \
               (env.current_player == 1 and not agent_plays_first):
                action = agent.select_action(state, available_actions)
            else:
                action = random.choice(available_actions)

            state, reward = env.step(action)
            steps += 1

        score = env.get_score()
        if score > 0:
            winner = env.current_player
            agent_won = (winner == 0 and agent_plays_first) or \
                        (winner == 1 and not agent_plays_first)
            total_score += 1 if agent_won else -1

        total_steps += steps

    avg_score = total_score / num_games
    avg_steps = total_steps / num_games
    return avg_score, avg_steps


def run_test(config, resume_from=None):
    env = Quatro(seed=42)
    eval_env = Quatro(seed=123)

    agent = ExpertApprentice(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        num_simulations=config["num_simulations"],
        buffer_size=config["buffer_size"],
    )

    if resume_from is not None:
        agent.load(resume_from)
        print(f"Reprise depuis {resume_from}")

    tracker = RLTracker(
        agent_name="ExIt",
        env_name="Quarto",
        config=config,
        run_name=f"{config['name']}_{config['total_games']}g",
    )

    total_games = config["total_games"]
    games_per_batch = config["games_per_batch"]
    eval_every = config.get("eval_every", games_per_batch)
    games_done = 0

    num_workers = config.get("num_workers", 1)
    print(f"Training ExIt on Quarto ({config['name']})...")
    print(f"  MCTS sims={config['num_simulations']}, "
          f"batch={games_per_batch}, total={total_games}, "
          f"workers={num_workers}")

    while games_done < total_games:
        batch_size = min(games_per_batch, total_games - games_done)

        # Collecte MCTS
        t0 = time.time()
        if num_workers > 1:
            agent.collect_games_parallel(env, batch_size, num_workers)
        else:
            agent.collect_games(env, batch_size)
        collect_time = time.time() - t0

        games_done += batch_size

        # Entraînement réseau
        loss = agent.train_network(
            epochs=config["train_epochs"],
            batch_size=config["train_batch_size"],
        )

        tracker.log_episode(games_done, score=0, loss=loss)

        print(f"  [{games_done}/{total_games}] "
              f"buffer={len(agent.buffer)}  loss={loss:.4f}  "
              f"collect={collect_time:.1f}s")

        # Évaluation + checkpoint tous les eval_every games
        if games_done % eval_every == 0 or games_done >= total_games:
            avg_score, avg_steps = evaluate_vs_random(
                agent, eval_env, config["eval_games"]
            )
            tracker.log_evaluation(games_done, avg_score=avg_score,
                                   avg_steps=avg_steps)

            # Temps par coup du réseau
            move_times = []
            eval_env.reset()
            for _ in range(200):
                eval_env.reset()
                while not eval_env.is_terminal():
                    avail = eval_env.get_available_actions()
                    if not avail:
                        break
                    t = time.time()
                    agent.select_action(eval_env.encode_state(), avail)
                    move_times.append(time.time() - t)
                    eval_env.step(random.choice(avail))

            avg_move_time = sum(move_times) / len(move_times) if move_times else 0
            tracker.log_move_time(avg_move_time)

            os.makedirs("models", exist_ok=True)
            path = f"models/exit_quarto_{games_done}g.pt"
            agent.save(path)

            print(f"  [Eval @ {games_done}] score={avg_score:.3f}  "
                  f"steps={avg_steps:.1f}  "
                  f"move_time={avg_move_time*1000:.3f}ms  saved={path}")

    # Évaluation finale
    avg_score, avg_steps = evaluate_vs_random(
        agent, eval_env, config["eval_games"]
    )
    tracker.log_evaluation(total_games, avg_score=avg_score, avg_steps=avg_steps)
    print(f"  [Final] score={avg_score:.3f}  steps={avg_steps:.1f}")

    tracker.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin vers un checkpoint .pt pour reprendre")
    args = parser.parse_args()

    for config in TEST_CONFIGS:
        print(f"\n{'='*50}")
        print(f"=== {config['name']} ===")
        print(f"{'='*50}")
        run_test(config, resume_from=args.resume)

    print("\n=== DONE ===")
