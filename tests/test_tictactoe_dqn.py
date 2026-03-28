import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.tic_tac_toe import TicTacToe
from agents.deep_qlearning import DQNAgent


TEST_CONFIGS = [
    {
        "name": "config_1",
        "state_size": 9,
        "action_size": 9,
        "hidden_dim": 128,
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "buffer_capacity": 10000,
        "batch_size": 64,
        "target_update_freq": 20,
        "num_episodes": 5000,
        "eval_episodes": 100,
    },
    {
        "name": "config_2",
        "state_size": 9,
        "action_size": 9,
        "hidden_dim": 64,
        "lr": 5e-4,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.995,
        "buffer_capacity": 10000,
        "batch_size": 64,
        "target_update_freq": 20,
        "num_episodes": 5000,
        "eval_episodes": 100,
    },
    {
        "name": "config_3",
        "state_size": 9,
        "action_size": 9,
        "hidden_dim": 128,
        "lr": 1e-4,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "buffer_capacity": 10000,
        "batch_size": 64,
        "target_update_freq": 10,
        "num_episodes": 5000,
        "eval_episodes": 100,
    },
]


def run_test(config):
    env = TicTacToe(seed=42)

    agent = DQNAgent(
        state_size=config["state_size"],
        action_size=config["action_size"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        epsilon_min=config["epsilon_min"],
        epsilon_decay=config["epsilon_decay"],
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
    )

    train_scores, train_losses = agent.train(env, num_episodes=config["num_episodes"])
    avg_score = agent.evaluate(env, num_episodes=config["eval_episodes"])

    avg_train_score = sum(train_scores) / len(train_scores) if train_scores else None
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else None

    return {
        "config_name": config["name"],
        "hidden_dim": config["hidden_dim"],
        "lr": config["lr"],
        "gamma": config["gamma"],
        "epsilon_start": config["epsilon"],
        "epsilon_min": config["epsilon_min"],
        "epsilon_decay": config["epsilon_decay"],
        "buffer_capacity": config["buffer_capacity"],
        "batch_size": config["batch_size"],
        "target_update_freq": config["target_update_freq"],
        "num_episodes": config["num_episodes"],
        "eval_episodes": config["eval_episodes"],
        "avg_train_score": avg_train_score,
        "avg_train_loss": avg_train_loss,
        "avg_score": avg_score,
        "final_epsilon": agent.epsilon,
    }


def save_results(results, file_path="results/tictactoe_dqn_results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)


if __name__ == "__main__":
    for config in TEST_CONFIGS:
        print(f"\n=== Running {config['name']} ===")

        results = run_test(config)

        for key, value in results.items():
            print(f"{key}: {value}")

        save_results(results)

    print("\n=== DONE ===")