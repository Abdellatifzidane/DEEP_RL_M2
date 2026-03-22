import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


RESULTS_DIR = "results"

RANDOM_FILE = os.path.join(RESULTS_DIR, "gridworld_random_results.csv")
TABULAR_FILE = os.path.join(RESULTS_DIR, "gridworld_tabular_q_learning_results.csv")
DQN_FILE = os.path.join(RESULTS_DIR, "gridworld_dqn_results.csv")


def load_results():
    data = []

    if os.path.exists(RANDOM_FILE):
        df_random = pd.read_csv(RANDOM_FILE)
        df_random["agent"] = "Random"
        data.append(df_random)

    if os.path.exists(TABULAR_FILE):
        df_tabular = pd.read_csv(TABULAR_FILE)
        df_tabular["agent"] = "Tabular Q-Learning"
        data.append(df_tabular)

    if os.path.exists(DQN_FILE):
        df_dqn = pd.read_csv(DQN_FILE)
        df_dqn["agent"] = "DQN"
        data.append(df_dqn)

    if not data:
        raise FileNotFoundError("Aucun fichier de résultats trouvé dans le dossier results/.")

    return pd.concat(data, ignore_index=True)


def build_summary(df):
    summary_rows = []

    for agent_name, group in df.groupby("agent"):
        avg_score = group["avg_score"].mean() if "avg_score" in group.columns else None

        row = {
            "agent": agent_name,
            "mean_avg_score": avg_score,
            "num_tests": len(group),
        }

        if "avg_reward" in group.columns:
            row["mean_avg_reward"] = group["avg_reward"].mean()

        if "avg_train_loss" in group.columns:
            row["mean_avg_train_loss"] = group["avg_train_loss"].dropna().mean()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def save_summary(summary_df, output_path=os.path.join(RESULTS_DIR, "summary_results.csv")):
    summary_df.to_csv(output_path, index=False)


def plot_avg_score(summary_df, output_path=os.path.join(RESULTS_DIR, "plot_avg_score.png")):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["agent"], summary_df["mean_avg_score"])
    plt.xlabel("Agent")
    plt.ylabel("Average Score")
    plt.title("Comparison of Agents on GridWorld")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_all_test_scores(df, output_path=os.path.join(RESULTS_DIR, "plot_all_test_scores.png")):
    labels = []
    values = []

    for _, row in df.iterrows():
        if row["agent"] == "Random":
            label = f"Random\n{row.get('test_name', 'baseline')}"
        else:
            label = f"{row['agent']}\n{row.get('config_name', 'config')}"

        labels.append(label)
        values.append(row["avg_score"])

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xlabel("Test")
    plt.ylabel("Average Score")
    plt.title("Score by Test Configuration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def main():
    df = load_results()

    print("\n=== RAW RESULTS ===")
    print(df)

    summary_df = build_summary(df)

    print("\n=== SUMMARY ===")
    print(summary_df)

    save_summary(summary_df)
    plot_avg_score(summary_df)
    plot_all_test_scores(df)


if __name__ == "__main__":
    main()