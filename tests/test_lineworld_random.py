import sys
import os
import csv
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.line_world import LineWorld
from agents.random import RandomAgent
from evaluate.tracker import RLTracker


# =========================
# CONFIGURATION (facile à modifier)
# =========================
CONFIG = {
    "test_name": "random_baseline",
    "num_episodes": 50,
}


# =========================
# TEST
# =========================
def run_test(config):
    env = LineWorld()
    agent = RandomAgent()

    # --- RLTracker ---
    tracker = RLTracker(
        agent_name="Random",
        env_name="LineWorld",
        config=config,
        run_name=config["test_name"],
    )

    scores = []
    total_rewards = []

    start_time = time.time()

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

        tracker.log_episode(episode, score=score)

    end_time = time.time()
    execution_time = end_time - start_time

    # métriques simples
    avg_score = sum(scores) / len(scores)
    avg_reward = sum(total_rewards) / len(total_rewards)

    tracker.log_evaluation(config["num_episodes"], avg_score=avg_score)
    tracker.log_move_time(execution_time / max(len(scores), 1))
    tracker.finish()

    return {
        "test_name": config["test_name"],
        "num_episodes": config["num_episodes"],
        "avg_score": avg_score,
        "avg_reward": avg_reward,
        "execution_time_sec": execution_time,
    }


# =========================
# SAVE RESULTS (CSV)
# =========================
def save_results(results, file_path="results/lineworld_random_results.csv"):
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
