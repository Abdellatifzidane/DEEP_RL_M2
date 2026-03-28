import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import os

from environnements.test_env.tic_tac_toe import TicTacToe
from agents.random import RandomAgent


# =========================
# CONFIGURATION (facile à modifier)
# =========================
CONFIG = {
    "test_name": "random_baseline",
    "num_episodes": 100,
}


# =========================
# TEST
# =========================
def run_test(config):
    env = TicTacToe(seed=42)
    agent = RandomAgent()

    scores = []
    total_rewards = []

    for episode in range(config["num_episodes"]):
        state = env.reset()
        episode_reward = 0

        while not env.is_terminal():
            action = agent.choose_action(env)
            next_state, reward = env.step(action)

            state = next_state
            episode_reward += reward

        score = env.get_score()
        scores.append(score)
        total_rewards.append(episode_reward)

    # métriques simples
    avg_score = sum(scores) / len(scores)
    avg_reward = sum(total_rewards) / len(total_rewards)

    return {
        "test_name": config["test_name"],
        "num_episodes": config["num_episodes"],
        "avg_score": avg_score,
        "avg_reward": avg_reward,
    }


# =========================
# SAVE RESULTS (CSV)
# =========================
def save_results(results, file_path="results/tictactoe_random_results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    results = run_test(CONFIG)

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    save_results(results)