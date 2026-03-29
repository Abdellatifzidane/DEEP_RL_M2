import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = "results"

RANDOM_FILE = os.path.join(RESULTS_DIR, "gridworld_random_results.csv")
TABULAR_FILE = os.path.join(RESULTS_DIR, "gridworld_tabular_q_learning_results.csv")
DQN_FILE = os.path.join(RESULTS_DIR, "gridworld_dqn_results.csv")


# =========================
# LOAD RESULTS
# =========================
def load_results():
    data = []

    if os.path.exists(RANDOM_FILE):
        df_random = pd.read_csv(RANDOM_FILE)
        df_random["agent"] = "Random"

        # harmonisation des colonnes
        if "test_name" in df_random.columns and "config_name" not in df_random.columns:
            df_random["config_name"] = df_random["test_name"]

        if "execution_time_sec" not in df_random.columns:
            df_random["execution_time_sec"] = None

        if "num_episodes" not in df_random.columns:
            df_random["num_episodes"] = None

        data.append(df_random)

    if os.path.exists(TABULAR_FILE):
        df_tabular = pd.read_csv(TABULAR_FILE)
        df_tabular["agent"] = "Tabular Q-Learning"

        if "execution_time_sec" not in df_tabular.columns:
            df_tabular["execution_time_sec"] = None

        data.append(df_tabular)

    if os.path.exists(DQN_FILE):
        df_dqn = pd.read_csv(DQN_FILE)
        df_dqn["agent"] = "DQN"

        if "execution_time_sec" not in df_dqn.columns:
            df_dqn["execution_time_sec"] = None

        data.append(df_dqn)

    if not data:
        raise FileNotFoundError("Aucun fichier de résultats trouvé dans le dossier results/.")

    return pd.concat(data, ignore_index=True)


# =========================
# FIND BEST CONFIG PER AGENT
# =========================
def find_best_configs(df):
    required_cols = ["agent", "avg_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne obligatoire manquante : {col}")

    # supprime les lignes sans avg_score
    df = df.dropna(subset=["avg_score"]).copy()

    # prend la ligne ayant le score max pour chaque agent
    best_idx = df.groupby("agent")["avg_score"].idxmax()
    best_df = df.loc[best_idx].copy()

    # colonnes utiles
    columns_to_keep = [
        "agent",
        "config_name",
        "num_episodes",
        "avg_score",
        "execution_time_sec",
    ]

    # garder seulement les colonnes existantes
    existing_cols = [col for col in columns_to_keep if col in best_df.columns]
    best_df = best_df[existing_cols]

    return best_df.sort_values("avg_score", ascending=False).reset_index(drop=True)


# =========================
# SAVE SUMMARY TABLE
# =========================
def save_best_summary(best_df, output_path=os.path.join(RESULTS_DIR, "best_agents_summary.csv")):
    best_df.to_csv(output_path, index=False)
    print(f"\nTableau récapitulatif sauvegardé : {output_path}")


# =========================
# PLOT BEST AVG SCORE
# =========================
def plot_best_scores(best_df, output_path=os.path.join(RESULTS_DIR, "best_agents_avg_score.png")):
    plt.figure(figsize=(8, 5))
    plt.bar(best_df["agent"], best_df["avg_score"])
    plt.xlabel("Agent")
    plt.ylabel("Average Score")
    plt.title("Best Configuration: Average Score by Agent")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print(f"Graphique score sauvegardé : {output_path}")


# =========================
# PLOT EXECUTION TIME
# =========================
def plot_execution_time(best_df, output_path=os.path.join(RESULTS_DIR, "best_agents_execution_time.png")):
    # on ne garde que les lignes avec temps connu
    plot_df = best_df.dropna(subset=["execution_time_sec"]).copy()

    if plot_df.empty:
        print("Aucun temps d'exécution disponible pour le graphique.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["agent"], plot_df["execution_time_sec"])
    plt.xlabel("Agent")
    plt.ylabel("Execution Time (sec)")
    plt.title("Best Configuration: Execution Time by Agent")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print(f"Graphique temps sauvegardé : {output_path}")


# =========================
# PRINT CLEAN SUMMARY
# =========================
def print_summary(best_df):
    print("\n=== BEST CONFIGURATION PER AGENT ===")
    print(best_df.to_string(index=False))


# =========================
# MAIN
# =========================
def main():
    df = load_results()

    print("\n=== RAW RESULTS ===")
    print(df.head())

    best_df = find_best_configs(df)

    print_summary(best_df)
    save_best_summary(best_df)
    plot_best_scores(best_df)
    plot_execution_time(best_df)


if __name__ == "__main__":
    main()