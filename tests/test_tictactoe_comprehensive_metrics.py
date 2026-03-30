"""
Comprehensive policy metrics evaluation on TicTacToe environment.
Tests all agents (Random, Tabular Q-Learning, DQN) with configs 1-3.
Measures policy performance at different training milestones.
"""

import csv
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.tic_tac_toe import TicTacToe
from agents.tabular_qlearning import TabularQLearningAgent
from agents.deep_qlearning import DQNAgent
from agents.random import RandomAgent


# Configurations for each agent
AGENT_CONFIGS = {
    "random": [
        {
            "name": "config_1",
            # Random agent has no hyperparameters
        }
    ],
    "tabular_qlearning": [
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
    ],
    "dqn": [
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
        },
    ],
}

# Training episode milestones per agent
MILESTONES = {
    "random": [1000],  # No actual training for random, just fixed evaluation
    "tabular_qlearning": [1000, 10000, 100000, 1000000],  # Tabular Q-learning is fast enough for 1M
    "dqn": [1000, 10000, 100000],  # DQN is slower, skip 1M to avoid excessive computation
}

EVAL_EPISODES = 100  # Number of episodes for policy evaluation


def evaluate_random_agent(env, agent, num_episodes):
    """Evaluate random agent (no training needed)."""
    scores = []
    for _ in range(num_episodes):
        env.reset()
        while not env.is_terminal():
            action = agent.choose_action(env)
            env.step(action)
        scores.append(env.get_score())
    return sum(scores) / len(scores) if scores else None


def run_comprehensive_metrics():
    """Run comprehensive policy evaluation across all agents and configurations."""
    env = TicTacToe(seed=42)
    results = []

    for agent_name in ["random", "tabular_qlearning", "dqn"]:
        print(f"\n{'='*60}")
        print(f"Testing {agent_name.upper()}")
        print(f"{'='*60}")

        for config in AGENT_CONFIGS[agent_name]:
            config_name = config["name"]
            print(f"\n--- Config: {config_name} ---")

            # Extract parameters for agent initialization
            config_params = {k: v for k, v in config.items() if k != "name"}

            # Initialize agent
            if agent_name == "random":
                agent = RandomAgent()
            elif agent_name == "tabular_qlearning":
                agent = TabularQLearningAgent(**config_params)
            elif agent_name == "dqn":
                agent = DQNAgent(**config_params)

            cumulative_episodes = 0

            for milestone in MILESTONES[agent_name]:
                episodes_to_train = milestone - cumulative_episodes

                # Train if needed (not for random agent)
                if agent_name != "random" and episodes_to_train > 0:
                    print(f"  Training {episodes_to_train} episodes (total: {milestone})...", end="", flush=True)
                    start_time = time.time()
                    agent.train(env, num_episodes=episodes_to_train)
                    train_time = time.time() - start_time
                    print(f" Done ({train_time:.2f}s)")

                # Evaluate learned policy
                print(
                    f"  Evaluating policy after {milestone} training episodes...",
                    end="",
                    flush=True,
                )
                start_time = time.time()
                if agent_name == "random":
                    avg_score = evaluate_random_agent(env, agent, EVAL_EPISODES)
                else:
                    avg_score = agent.evaluate(env, num_episodes=EVAL_EPISODES)
                eval_time = time.time() - start_time

                print(f" Score: {avg_score:.4f} ({eval_time:.2f}s)")

                result = {
                    "agent": agent_name,
                    "config": config_name,
                    "training_episodes": milestone,
                    "policy_avg_score": avg_score,
                    "eval_episodes": EVAL_EPISODES,
                }
                results.append(result)
                
                # Save immediately to avoid data loss
                append_result(result)

                cumulative_episodes = milestone

    return results


def save_results(results, file_path="results/tictactoe_comprehensive_metrics.csv"):
    """Save results to CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        if results:
            fieldnames = [
                "agent",
                "config",
                "training_episodes",
                "policy_avg_score",
                "eval_episodes",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n✓ Results saved to {file_path}")


def append_result(result, file_path="results/tictactoe_comprehensive_metrics.csv"):
    """Append a single result to CSV file immediately (for incremental saving)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)
    fieldnames = [
        "agent",
        "config",
        "training_episodes",
        "policy_avg_score",
        "eval_episodes",
    ]

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def display_results_table(results):
    """Display results in a formatted table."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE METRICS SUMMARY")
    print(f"{'='*80}")

    # Group by agent
    agents = {}
    for result in results:
        agent = result["agent"]
        if agent not in agents:
            agents[agent] = {}
        config = result["config"]
        if config not in agents[agent]:
            agents[agent][config] = []
        agents[agent][config].append(result)

    for agent in ["random", "tabular_qlearning", "dqn"]:
        if agent not in agents:
            continue
        print(f"\n{agent.upper()}")
        print("-" * 80)
        print(
            f"{'Config':<12} {'Training Episodes':<20} {'Policy Avg Score':<20} {'Eval Episodes':<15}"
        )
        print("-" * 80)

        for config in sorted(agents[agent].keys()):
            for result in agents[agent][config]:
                print(
                    f"{result['config']:<12} {result['training_episodes']:<20} {result['policy_avg_score']:<20.4f} {result['eval_episodes']:<15}"
                )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TICTACTOE COMPREHENSIVE POLICY METRICS EVALUATION")
    print("=" * 80)
    print(f"Evaluation episodes per checkpoint: {EVAL_EPISODES}")
    print(f"Environment seed: 42")

    results = run_comprehensive_metrics()
    display_results_table(results)

    print(f"\n✓ Evaluation complete! Total results: {len(results)}")
