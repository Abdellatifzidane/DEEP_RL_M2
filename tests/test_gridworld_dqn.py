import csv
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.grid_world import GridWorld
from agents.deep_qlearning import DQNAgent


# =========================
# CONFIGS DQN
# =========================
BASE_CONFIGS = [
    {
        "name": "config_2",
        "grid_size": 5,
        "state_size": 2,
        "action_size": 4,
        "hidden_dim": 64,
        "lr": 5e-4,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.995,
        "buffer_capacity": 10000,
        "batch_size": 64,
        "target_update_freq": 20,
        "eval_episodes": 50,
    },
    {
        "name": "config_3",
        "grid_size": 5,
        "state_size": 2,
        "action_size": 4,
        "hidden_dim": 128,
        "lr": 1e-4,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "buffer_capacity": 10000,
        "batch_size": 64,
        "target_update_freq": 10,
        "eval_episodes": 50,
    },
]

# =========================
# NOMBRE D'EPISODES A TESTER
# =========================
EPISODE_CONFIGS = [1000, 10000]


# =========================
# RUN TEST
# =========================
def run_test(base_config, num_episodes):
    env = GridWorld(size=base_config["grid_size"])

    agent = DQNAgent(
        state_size=base_config["state_size"],
        action_size=base_config["action_size"],
        hidden_dim=base_config["hidden_dim"],
        lr=base_config["lr"],
        gamma=base_config["gamma"],
        epsilon=base_config["epsilon"],
        epsilon_min=base_config["epsilon_min"],
        epsilon_decay=base_config["epsilon_decay"],
        buffer_capacity=base_config["buffer_capacity"],
        batch_size=base_config["batch_size"],
        target_update_freq=base_config["target_update_freq"],
    )
    #  START TIMER
    start_time = time.time()
    train_scores, train_losses = agent.train(env, num_episodes=num_episodes)
    avg_score = agent.evaluate(env, num_episodes=base_config["eval_episodes"])
    #  END TIMER
    end_time = time.time()
    execution_time = end_time - start_time
    avg_train_score = sum(train_scores) / len(train_scores) if train_scores else None
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else None

    return {
        "config_name": base_config["name"],
        "grid_size": base_config["grid_size"],
        "hidden_dim": base_config["hidden_dim"],
        "lr": base_config["lr"],
        "gamma": base_config["gamma"],
        "epsilon_start": base_config["epsilon"],
        "epsilon_min": base_config["epsilon_min"],
        "epsilon_decay": base_config["epsilon_decay"],
        "buffer_capacity": base_config["buffer_capacity"],
        "batch_size": base_config["batch_size"],
        "target_update_freq": base_config["target_update_freq"],
        "num_episodes": num_episodes,
        "eval_episodes": base_config["eval_episodes"],
        "avg_train_score": avg_train_score,
        "avg_train_loss": avg_train_loss,
        "avg_score": avg_score,
        "final_epsilon": agent.epsilon,
        "execution_time_sec": execution_time,
    }


# =========================
# SAVE RESULTS
# =========================
def save_results(results_list, file_path="results/gridworld_dqn_results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not results_list:
        return

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    all_results = []

    for base_config in BASE_CONFIGS:
        for num_episodes in EPISODE_CONFIGS:
            print(f"\n=== Running {base_config['name']} with {num_episodes} episodes ===")

            results = run_test(base_config, num_episodes)
            all_results.append(results)

            for key, value in results.items():
                if key == "execution_time_sec":
                    print(f"{key}: {value:.2f} sec")
                else:
                    print(f"{key}: {value}")

    save_results(all_results)

    print("\n=== DONE ===")
