import csv
import os
import sys
import time

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.tic_tac_toe import TicTacToe
from agents.tabular_qlearning import TabularQLearningAgent
from agents.deep_qlearning import DQNAgent
from agents.random import RandomAgent


# Configurations for each agent
AGENT_CONFIGS = {
    "random": [
        {"name": "config_1"}  # Only one for random
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
            "batch_size": 16,  # Reduced for speed
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
            "batch_size": 16,
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
            "batch_size": 16,
            "target_update_freq": 10,
        },
    ]
}

# Training episode milestones per agent
MILESTONES = {
    "random": [0],  # No training
    "tabular_qlearning": [1000, 10000, 100000],
    "dqn": [1000, 10000]
}

EVAL_EPISODES = 10  # Number of episodes for evaluation


def evaluate_random_agent(env, agent, num_episodes):
    """Evaluate random agent (no training needed)"""
    scores = []
    for _ in range(num_episodes):
        env.reset()
        while not env.is_terminal():
            action = agent.choose_action(env)
            env.step(action)
        scores.append(env.get_score())
    return sum(scores) / len(scores) if scores else None


def run_metrics():
    env = TicTacToe(seed=42)
    results = []

    for agent_name in AGENT_CONFIGS:
        print(f"\n=== Testing {agent_name.upper()} ===")
        for config in AGENT_CONFIGS[agent_name]:
            print(f"Config: {config['name']}")
            config_params = {k: v for k, v in config.items() if k != 'name'}
            
            if agent_name == "random":
                agent = RandomAgent()
            elif agent_name == "tabular_qlearning":
                agent = TabularQLearningAgent(**config_params)
            elif agent_name == "dqn":
                agent = DQNAgent(**config_params)
            
            cumulative_episodes = 0
            for milestone in MILESTONES[agent_name]:
                episodes_to_train = milestone - cumulative_episodes
                if episodes_to_train > 0:
                    print(f"Training for {episodes_to_train} more episodes (total: {milestone})...")
                    start_time = time.time()
                    if agent_name != "random":
                        agent.train(env, num_episodes=episodes_to_train)
                    train_time = time.time() - start_time
                    if agent_name != "random":
                        print(".2f")
                    cumulative_episodes = milestone

                print(f"Evaluating after {milestone} training episodes...")
                if agent_name == "random":
                    avg_score = evaluate_random_agent(env, agent, EVAL_EPISODES)
                else:
                    avg_score = agent.evaluate(env, num_episodes=EVAL_EPISODES)
                
                results.append({
                    "agent": agent_name,
                    "config": config['name'],
                    "training_episodes": milestone,
                    "avg_score": avg_score,
                    "eval_episodes": EVAL_EPISODES
                })
                print(f"{agent_name} - {config['name']} - {milestone} episodes - Avg Score: {avg_score}")

    return results


def save_results(results, file_path="results/tictactoe_policy_metrics_configs.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="") as file:
        if results:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    print("Running TicTacToe Policy Metrics Evaluation...")
    results = run_metrics()
    save_results(results)
    print("Results saved to results/tictactoe_policy_metrics_configs.csv")
    print("\nFinal Results:")
    for result in results:
        print(f"{result['agent']} - {result['config']} - {result['training_episodes']} episodes - Avg Score: {result['avg_score']}")