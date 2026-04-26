"""
Test PPO-A2C sur Quarto.

Boucle : jouer vs random → stocker rewards → update PPO → évaluer aux paliers.
"""

import os
import sys
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.quarto.quatro import Quatro, ACTION_SIZE, STATE_SIZE
from agents.ppo_a2c import PPO_A2C
from evaluate.tracker import RLTracker
import numpy as np
import torch


TEST_CONFIGS = [
    {
        "name": "ppo_100k",
        "state_dim": STATE_SIZE,
        "action_dim": ACTION_SIZE,
        "hidden_dim": 256,
        "lr": 3e-4,
        "gamma": 0.95,
        "eps_clip": 0.1,
        "k_epochs": 8,
        "batch_size": 128,
        "num_episodes": 100_000,
        "update_interval": 10,
        "eval_games": 100,
        "eval_every": 1_000,
    },
]


def evaluate_vs_random(agent, env, num_games=100):
    """Évalue la policy apprise (greedy) contre un joueur random.
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
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, _ = agent.policy(state_tensor)
                    probs = action_probs[0, available_actions].numpy()
                    probs = probs / probs.sum()
                    action = available_actions[np.argmax(probs)]
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

    return total_score / num_games, total_steps / num_games


def run_test(config, resume_from=None):
    env = Quatro(seed=42)
    eval_env = Quatro(seed=123)

    agent = PPO_A2C(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        eps_clip=config["eps_clip"],
        k_epochs=config["k_epochs"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
    )

    if resume_from is not None:
        agent.load(resume_from)
        print(f"Reprise depuis {resume_from}")

    tracker = RLTracker(
        agent_name="PPO_A2C",
        env_name="Quarto",
        config=config,
        run_name=f"{config['name']}_{config['num_episodes']}ep",
    )

    episodes = config["num_episodes"]
    update_interval = config["update_interval"]
    eval_every = config.get("eval_every", 1_000)

    print(f"Training PPO-A2C on Quarto ({config['name']})...")
    print(f"  episodes={episodes}, update_interval={update_interval}, "
          f"eval_every={eval_every}")

    for episode in range(episodes):
        state = env.reset()
        steps = 0
        agent_is_p0 = episode % 2 == 0

        while not env.is_terminal():
            available_actions = env.get_available_actions()
            if not available_actions:
                break

            is_agent_turn = (env.current_player == 0 and agent_is_p0) or \
                            (env.current_player == 1 and not agent_is_p0)

            if is_agent_turn:
                action = agent.select_action(state, available_actions)
            else:
                action = random.choice(available_actions)

            next_state, reward = env.step(action)

            if is_agent_turn:
                agent.store_reward(reward, env.is_terminal())
            elif env.is_terminal() and len(agent.memory['rewards']) > 0:
                agent.memory['rewards'][-1] = -1.0
                agent.memory['is_terminals'][-1] = True

            state = next_state
            steps += 1

        # Score du point de vue de l'agent
        agent_score = 0
        if env.get_score() > 0:
            winner = env.current_player
            agent_won = (winner == 0 and agent_is_p0) or \
                        (winner == 1 and not agent_is_p0)
            agent_score = 1 if agent_won else -1

        if (episode + 1) % update_interval == 0:
            agent.update()

        tracker.log_episode(episode, score=agent_score, steps=steps)

        if (episode + 1) % 1000 == 0:
            print(f"  Episode {episode + 1}/{episodes}")

        # Évaluation + checkpoint tous les eval_every épisodes
        if (episode + 1) % eval_every == 0:
            avg_score, avg_steps = evaluate_vs_random(
                agent, eval_env, config["eval_games"]
            )
            tracker.log_evaluation(episode + 1, avg_score=avg_score,
                                   avg_steps=avg_steps)

            # Temps par coup du réseau
            move_times = []
            for _ in range(200):
                eval_env.reset()
                while not eval_env.is_terminal():
                    avail = eval_env.get_available_actions()
                    if not avail:
                        break
                    t = time.time()
                    with torch.no_grad():
                        st = torch.FloatTensor(eval_env.encode_state()).unsqueeze(0)
                        ap, _ = agent.policy(st)
                        probs = ap[0, avail].numpy()
                        _ = avail[np.argmax(probs)]
                    move_times.append(time.time() - t)
                    eval_env.step(random.choice(avail))

            avg_move_time = sum(move_times) / len(move_times) if move_times else 0
            tracker.log_move_time(avg_move_time)

            os.makedirs("models", exist_ok=True)
            path = f"models/ppo_quarto_{episode+1}ep.pt"
            agent.save(path)

            print(f"  [Eval @ {episode+1}] score={avg_score:.3f}  "
                  f"steps={avg_steps:.1f}  "
                  f"move_time={avg_move_time*1000:.3f}ms  saved={path}")

    # Évaluation finale
    avg_score, avg_steps = evaluate_vs_random(
        agent, eval_env, config["eval_games"]
    )
    tracker.log_evaluation(episodes, avg_score=avg_score, avg_steps=avg_steps)
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
