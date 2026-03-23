import csv
import os
import sys

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.tic_tac_toe import TicTacToe
from agents.tabular_qlearning import TabularQLearningAgent


# =========================
# CONFIGURATIONS (hyperparamètres)
# =========================
TEST_CONFIGS = [
    {
        "name": "config_1",
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "num_episodes": 5000,
    },
    {
        "name": "config_2",
        "alpha": 0.05,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
        "num_episodes": 5000,
    },
    {
        "name": "config_3",
        "alpha": 0.2,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.98,
        "num_episodes": 5000,
    },
]


# =========================
# RUN TEST
# =========================
def run_test(config):
    env = TicTacToe(seed=42)

    agent = TabularQLearningAgent(
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        epsilon_min=config["epsilon_min"],
        epsilon_decay=config["epsilon_decay"],
    )

    # TRAIN
    training_scores = agent.train(env, num_episodes=config["num_episodes"])

    # EVALUATE
    avg_score = agent.evaluate(env, num_episodes=100)

    return {
        "config_name": config["name"],
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "epsilon_start": config["epsilon"],
        "epsilon_min": config["epsilon_min"],
        "epsilon_decay": config["epsilon_decay"],
        "num_episodes": config["num_episodes"],
        "avg_score": avg_score,
        "final_epsilon": agent.epsilon,
    }


# =========================
# SAVE RESULTS
# =========================
def save_results(results, file_path="results/tictactoe_tabular_q_learning_results.csv"):
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
    all_results = []

    for config in TEST_CONFIGS:
        print(f"\n=== Running {config['name']} ===")

        results = run_test(config)
        all_results.append(results)

        for k, v in results.items():
            print(f"{k}: {v}")

        save_results(results)

    print("\n=== DONE ===")