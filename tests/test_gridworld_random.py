import sys
import os
import csv
import time  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.grid_world import GridWorld
from agents.random import RandomAgent


# =========================
# CONFIGURATIONS DE TEST
# =========================
TEST_CONFIGS = [
    {
        "test_name": "random_1000_episodes",
        "grid_size": 5,
        "num_episodes": 1000,
    },
    {
        "test_name": "random_10000_episodes",
        "grid_size": 5,
        "num_episodes": 10000,
    },
    {
        "test_name": "random_100000_episodes",
        "grid_size": 5,
        "num_episodes": 100000,
    },
    {
        "test_name": "random_1000000_episodes",
        "grid_size": 5,
        "num_episodes": 1000000,
    },
]


# =========================
# TEST
# =========================
def run_test(config):
    env = GridWorld(size=config["grid_size"])
    agent = RandomAgent()

    scores = []
    total_rewards = []

    # ⏱️ START
    start_time = time.time()

    for episode in range(config["num_episodes"]):
        env.reset()
        episode_reward = 0

        while not env.is_terminal():
            action = agent.choose_action(env)
            _, reward = env.step(action)
            episode_reward += reward

        score = env.get_score()
        scores.append(score)
        total_rewards.append(episode_reward)

    # ⏱️ END
    end_time = time.time()
    execution_time = end_time - start_time

    avg_score = sum(scores) / len(scores)
    avg_reward = sum(total_rewards) / len(total_rewards)

    return {
        "test_name": config["test_name"],
        "agent": "RandomAgent",
        "grid_size": config["grid_size"],
        "num_episodes": config["num_episodes"],
        "avg_score": avg_score,
        "avg_reward": avg_reward,
        "execution_time_sec": execution_time,  
    }


# =========================
# SAVE RESULTS (CSV)
# =========================
def save_results(results_list, file_path="results/gridworld_random_results.csv"):
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

    for config in TEST_CONFIGS:
        print(f"\nRunning test: {config['test_name']} ...")
        results = run_test(config)
        all_results.append(results)

        print("=== RESULTS ===")
        for k, v in results.items():
            if k == "execution_time_sec":
                print(f"{k}: {v:.2f} sec")  
            else:
                print(f"{k}: {v}")

    save_results(all_results)
    print("\nTous les résultats ont été enregistrés dans results/gridworld_random_results.csv")
