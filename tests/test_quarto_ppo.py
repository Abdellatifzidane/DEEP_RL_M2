import csv
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.quarto.quatro import Quatro
from agents.ppo_a2c import PPO_A2C
from evaluate.tracker import RLTracker
import numpy as np
import torch
import random


TEST_CONFIGS = [
    {
        "name": "config_2",
        "state_dim": 68,
        "action_dim": 32,
        "hidden_dim": 256,
        "lr": 3e-4,
        "gamma": 0.95,
        "eps_clip": 0.1,
        "k_epochs": 8,
        "batch_size": 128,
        "num_episodes": 10000,
        "update_interval": 10,
        "eval_games": 50,
    },
]


def train_vs_random(agent, env, tracker, episodes=1000, update_interval=10):
    """Entraîne l'agent PPO contre un joueur random. L'agent est P0."""
    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
        state = env.reset()
        steps = 0
        agent_is_p0 = episode % 2 == 0  # alterner qui commence

        while not env.is_terminal():
            available_actions = env.get_available_actions()
            if len(available_actions) == 0:
                break

            is_agent_turn = (env.current_player == 0 and agent_is_p0) or \
                            (env.current_player == 1 and not agent_is_p0)

            if is_agent_turn:
                # L'agent PPO joue
                flat_actions = [a[0] * 16 + a[1] for a in available_actions]
                action_idx = agent.select_action(state, flat_actions)
                action = (action_idx // 16, action_idx % 16)
            else:
                # Le joueur random joue
                action = random.choice(available_actions)

            next_state, reward = env.step(action)

            # Ne stocker le reward que pour les coups de l'agent
            if is_agent_turn:
                agent.store_reward(reward, env.is_terminal())
            elif env.is_terminal() and len(agent.memory['rewards']) > 0:
                # Le random a gagné → punir le dernier coup de l'agent
                agent.memory['rewards'][-1] = -1.0
                agent.memory['is_terminals'][-1] = True

            state = next_state
            steps += 1

        final_score = env.get_score()
        if final_score > 0:
            winner = env.current_player
            agent_won = (winner == 0 and agent_is_p0) or \
                        (winner == 1 and not agent_is_p0)
            if agent_won:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

        if (episode + 1) % update_interval == 0:
            agent.update()

        # Log vers le tracker
        tracker.log_episode(
            episode,
            score=final_score,
            steps=steps,
            win_rate=wins / (episode + 1),
            loss_rate=losses / (episode + 1),
            draw_rate=draws / (episode + 1),
        )

        if (episode + 1) % 100 == 0:
            win_pct = wins / (episode + 1) * 100
            loss_pct = losses / (episode + 1) * 100
            draw_pct = draws / (episode + 1) * 100
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Wins: {win_pct:.1f}%, Losses: {loss_pct:.1f}%, Draws: {draw_pct:.1f}%")

    print("\n=== Training Complete ===")
    print(f"Final Win rate: {wins/episodes*100:.1f}%")
    print(f"Final Loss rate: {losses/episodes*100:.1f}%")
    print(f"Final Draw rate: {draws/episodes*100:.1f}%")

    return agent


def evaluate_vs_random(agent, env, num_games=50):
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0

    for game in range(num_games):
        state = env.reset()
        agent_plays_first = game % 2 == 0
        steps = 0

        while not env.is_terminal():
            available_actions = env.get_available_actions()
            if len(available_actions) == 0:
                break

            if (env.current_player == 0 and agent_plays_first) or \
               (env.current_player == 1 and not agent_plays_first):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, _ = agent.policy(state_tensor)

                    valid_actions = []
                    valid_probs = []
                    for action in available_actions:
                        action_type, idx = action
                        action_idx = action_type * 16 + idx
                        valid_actions.append(action)
                        valid_probs.append(action_probs[0, action_idx].item())

                    valid_probs = np.array(valid_probs)
                    valid_probs = valid_probs / valid_probs.sum()
                    action = valid_actions[np.argmax(valid_probs)]
            else:
                action = random.choice(available_actions)

            state, reward = env.step(action)
            steps += 1

        score = env.get_score()
        total_steps += steps
        if score > 0:
            # Le gagnant est env.current_player (celui qui a posé la pièce)
            winner = env.current_player
            agent_won = (winner == 0 and agent_plays_first) or \
                        (winner == 1 and not agent_plays_first)
            if agent_won:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    print("\n=== Evaluation vs Random Player ===")
    print(f"Games played: {num_games}")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

    avg_score = wins / num_games
    avg_steps = total_steps / num_games
    return avg_score, avg_steps, wins, losses, draws


def run_test(config):
    env = Quatro(seed=42)

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

    tracker = RLTracker(
        agent_name="PPO_A2C",
        env_name="Quarto",
        config=config,
        run_name=f"{config['name']}_{config['num_episodes']}ep",
    )

    start_time = time.time()

    print(f"Training PPO-A2C on Quarto vs Random ({config['name']})...")
    agent = train_vs_random(
        agent, env, tracker,
        episodes=config["num_episodes"],
        update_interval=config["update_interval"],
    )

    print("\nEvaluating trained agent vs random player...")
    eval_env = Quatro(seed=123)
    avg_score, avg_steps, wins, losses, draws = evaluate_vs_random(
        agent, eval_env, num_games=config["eval_games"]
    )

    end_time = time.time()
    execution_time = end_time - start_time

    tracker.log_evaluation(
        config["num_episodes"],
        avg_score=avg_score,
        avg_steps=avg_steps,
        win_rate=wins / config["eval_games"],
        loss_rate=losses / config["eval_games"],
        draw_rate=draws / config["eval_games"],
    )
    tracker.log_move_time(execution_time / config["num_episodes"])
    tracker.finish()

    return {
        "config_name": config["name"],
        "hidden_dim": config["hidden_dim"],
        "lr": config["lr"],
        "gamma": config["gamma"],
        "eps_clip": config["eps_clip"],
        "k_epochs": config["k_epochs"],
        "num_episodes": config["num_episodes"],
        "update_interval": config["update_interval"],
        "eval_win_rate": avg_score,
        "eval_wins": wins,
        "eval_losses": losses,
        "eval_draws": draws,
        "execution_time_sec": round(execution_time, 2),
    }


def save_results(results, file_path="results/quarto_ppo_results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


if __name__ == "__main__":
    for config in TEST_CONFIGS:
        print(f"\n{'='*50}")
        print(f"=== Running {config['name']} ===")
        print(f"{'='*50}")

        results = run_test(config)

        print("\n--- Results ---")
        for key, value in results.items():
            print(f"  {key}: {value}")

        save_results(results)

    print("\n=== ALL TESTS DONE ===")
