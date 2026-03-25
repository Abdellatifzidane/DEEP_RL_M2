"""
Test Quarto : deux RandomAgent s'affrontent via l'interface BaseEnv.
Chaque joueur appelle step() à son tour.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv

from environnements.quarto.quatro import Quatro
from agents.random import RandomAgent


# =========================
# CONFIGURATION
# =========================
CONFIG = {
    "test_name": "random_vs_random",
    "num_episodes": 1000,
}


# =========================
# TEST
# =========================
def run_test(config):
    player1 = RandomAgent()
    player2 = RandomAgent()
    players = [player1, player2]

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    for episode in range(config["num_episodes"]):
        env = Quatro()
        env.reset()

        while not env.is_terminal():
            current = players[env.current_player]
            action = current.choose_action(env)
            state, reward = env.step(action)

            if reward == 1.0:
                # Le joueur qui vient de poser a gagné
                # current_player a déjà basculé dans step(), le gagnant est l'autre
                winner = 1 - env.current_player
                if winner == 0:
                    wins_p1 += 1
                else:
                    wins_p2 += 1

        if env.get_score() == 0.0 and env.is_terminal():
            draws += 1

    total = config["num_episodes"]
    return {
        "test_name": config["test_name"],
        "num_episodes": total,
        "wins_player1": wins_p1,
        "wins_player2": wins_p2,
        "draws": draws,
        "win_rate_p1": round(wins_p1 / total, 3),
        "win_rate_p2": round(wins_p2 / total, 3),
    }


# =========================
# SAVE RESULTS (CSV)
# =========================
def save_results(results, file_path="results/quarto_random_random_results.csv"):
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
