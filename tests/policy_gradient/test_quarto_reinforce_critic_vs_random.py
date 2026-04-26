import os
import time
import random
import torch

from agents.policy_gradient.reinforce_critic import ReinforceCriticAgent
from environnements.quarto.quatro import Quatro
from evaluate.tracker import RLTracker


CONFIG = {
    "agent_name": "ReinforceCriticVsRandom",
    "env_name": "Quatro",

    "gamma": 0.99,
    "learning_rate_policy": 1e-3,
    "learning_rate_value": 1e-3,

    "eval_milestones": [1_000, 10_000, 100_000, 200_000, 300_000],
    "num_eval_episodes": 200,

    "checkpoint_dir": "checkpoints",
    "log_dir": "runs",
    "csv_dir": "results",
}


def load_agent(checkpoint_path):
    env = Quatro()

    taille_etat = len(env.get_state())
    nombre_actions = env.action_size

    agent = ReinforceCriticAgent(
        taille_etat=taille_etat,
        nombre_actions=nombre_actions,
        gamma=CONFIG["gamma"],
        learning_rate_policy=CONFIG["learning_rate_policy"],
        learning_rate_value=CONFIG["learning_rate_value"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    agent.policy_network.load_state_dict(checkpoint["policy_state_dict"])

    # Le critic est chargé aussi pour garder le checkpoint complet,
    # même si l'évaluation utilise surtout la policy.
    agent.value_network.load_state_dict(checkpoint["value_state_dict"])

    agent.policy_network.eval()
    agent.value_network.eval()

    return agent


def evaluate_vs_random(agent, env_class, num_eval_episodes):
    agent_wins = 0
    random_wins = 0
    draws = 0

    total_steps = 0
    total_moves = 0
    total_move_time = 0.0

    for episode_idx in range(num_eval_episodes):
        env = env_class()
        env.reset()

        agent_player_id = episode_idx % 2
        env.current_player = episode_idx % 2

        episode_steps = 0
        winner = None

        while not env.is_terminal():
            current_player = env.current_player
            actions = env.get_available_actions()

            if not actions:
                break

            start_time = time.time()

            if current_player == agent_player_id:
                action = agent.agir(env)
            else:
                action = random.choice(actions)

            end_time = time.time()

            total_move_time += end_time - start_time
            total_moves += 1

            if action is None:
                break

            acting_player = current_player
            _, reward = env.step(action)
            episode_steps += 1

            if reward == 1.0:
                if acting_player == agent_player_id:
                    winner = "agent"
                else:
                    winner = "random"
                break

        if winner == "agent":
            agent_wins += 1
        elif winner == "random":
            random_wins += 1
        else:
            draws += 1

        total_steps += episode_steps

    win_rate = agent_wins / num_eval_episodes
    loss_rate = random_wins / num_eval_episodes
    draw_rate = draws / num_eval_episodes
    avg_steps = total_steps / num_eval_episodes
    avg_move_time = total_move_time / total_moves if total_moves > 0 else 0.0

    return win_rate, loss_rate, draw_rate, avg_steps, avg_move_time


def main():
    tracker = RLTracker(
        agent_name=CONFIG["agent_name"],
        env_name=CONFIG["env_name"],
        config=CONFIG,
        log_dir=CONFIG["log_dir"],
        csv_dir=CONFIG["csv_dir"],
    )

    try:
        for milestone in CONFIG["eval_milestones"]:
            checkpoint_path = os.path.join(
                CONFIG["checkpoint_dir"],
                "quatro",
                "reinforcecritic",
                f"episode_{milestone}.pt"
            )

            if not os.path.exists(checkpoint_path):
                print(f"[SKIP] Checkpoint introuvable : {checkpoint_path}")
                continue

            agent = load_agent(checkpoint_path)

            start_total_time = time.time()

            win_rate, loss_rate, draw_rate, avg_steps, avg_move_time = evaluate_vs_random(
                agent=agent,
                env_class=Quatro,
                num_eval_episodes=CONFIG["num_eval_episodes"],
            )

            total_time = time.time() - start_total_time

            print(f"[DEBUG] LOGGED STEP = {milestone}")

            tracker.log_evaluation(
                num_episodes_trained=milestone,
                avg_score=win_rate,
                avg_steps=avg_steps,
                loss_rate=loss_rate,
                draw_rate=draw_rate,
                avg_move_time_sec=avg_move_time,
                total_eval_time_sec=total_time,
            )

            tracker.writer.flush()

            print(
                f"[EVAL VS RANDOM] Episode {milestone} | "
                f"win_rate={win_rate:.4f} | "
                f"loss_rate={loss_rate:.4f} | "
                f"draw_rate={draw_rate:.4f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"avg_move_time={avg_move_time:.6f}s | "
                f"total_time={total_time:.2f}s"
            )

    finally:
        tracker.finish()


if __name__ == "__main__":
    main()