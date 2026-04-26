import time
import random

from agents.simulation.randomrollout import RandomRolloutAgent
from environnements.quarto.quatro import Quatro


CONFIG = {
    "num_rollouts_list": [100, 500, 1000],
    "num_eval_episodes": 50,  # 🔥 important pour ne pas exploser le temps
}


def evaluate(agent, env_class, num_eval_episodes):
    agent_wins = 0
    random_wins = 0
    draws = 0

    total_steps = 0

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

            if current_player == agent_player_id:
                action = agent.act(env)
            else:
                action = random.choice(actions)

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

    return (
        agent_wins / num_eval_episodes,
        random_wins / num_eval_episodes,
        draws / num_eval_episodes,
        total_steps / num_eval_episodes,
    )


def main():
    for num_rollouts in CONFIG["num_rollouts_list"]:
        print("\n" + "=" * 60)
        print(f"TEST num_rollouts = {num_rollouts}")

        agent = RandomRolloutAgent(num_rollouts=num_rollouts)

        start = time.time()

        win_rate, loss_rate, draw_rate, avg_steps = evaluate(
            agent,
            Quatro,
            CONFIG["num_eval_episodes"]
        )

        total_time = time.time() - start

        print(
            f"RESULT | rollouts={num_rollouts} | "
            f"win_rate={win_rate:.4f} | "
            f"loss_rate={loss_rate:.4f} | "
            f"draw_rate={draw_rate:.4f} | "
            f"avg_steps={avg_steps:.2f} | "
            f"total_time={total_time:.2f}s"
        )


if __name__ == "__main__":
    main()