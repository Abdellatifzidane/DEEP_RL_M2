import csv
import os
import sys
import time  # 

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.grid_world import GridWorld
from agents.tabular_qlearning import TabularQLearningAgent


# =========================
# CONFIGURATIONS (hyperparamètres)
# =========================
BASE_CONFIGS = [
    {
        "name": "config_1",
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
    },
    {
        "name": "config_2",
        "alpha": 0.05,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
    },
    {
        "name": "config_3",
        "alpha": 0.2,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.98,
    },
]

# =========================
# NOMBRE D'ÉPISODES À TESTER
# =========================
EPISODE_CONFIGS = [1000, 10000, 100000, 1000000]


# =========================
# RUN TEST
# =========================
def run_test(base_config, num_episodes):
    env = GridWorld(size=5)

    agent = TabularQLearningAgent(
        alpha=base_config["alpha"],
        gamma=base_config["gamma"],
        epsilon=base_config["epsilon"],
        epsilon_min=base_config["epsilon_min"],
        epsilon_decay=base_config["epsilon_decay"],
    )

    #  START
    start_time = time.time()

    # TRAIN
    agent.train(env, num_episodes=num_episodes)

    # EVALUATE
    avg_score = agent.evaluate(env, num_episodes=50)

    #  END
    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "config_name": base_config["name"],
        "alpha": base_config["alpha"],
        "gamma": base_config["gamma"],
        "epsilon_start": base_config["epsilon"],
        "epsilon_min": base_config["epsilon_min"],
        "epsilon_decay": base_config["epsilon_decay"],
        "num_episodes": num_episodes,
        "avg_score": avg_score,
        "final_epsilon": agent.epsilon,
        "execution_time_sec": execution_time,  
    }


# =========================
# SAVE RESULTS
# =========================
def save_results(results_list, file_path="results/gridworld_tabular_q_learning_results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not results_list:
        return

    with open(file_path, mode="w", newline="") as file:
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

            for k, v in results.items():
                if k == "execution_time_sec":
                    print(f"{k}: {v:.2f} sec")  
                else:
                    print(f"{k}: {v}")

    save_results(all_results)

    print("\n=== DONE ===")
