import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environnements.test_env.tic_tac_toe import TicTacToe
from agents.ppo_a2c import PPO_A2C
import numpy as np
import torch


def train_ppo_tictactoe(episodes=1000, update_interval=10):
    env = TicTacToe(seed=42)

    state_dim = 9
    action_dim = 9

    agent = PPO_A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        batch_size=32,
        hidden_dim=64
    )

    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while not env.is_terminal():
            available_actions = env.get_available_actions()

            if len(available_actions) == 0:
                break

            action = agent.select_action(state, available_actions)

            next_state, reward = env.step(action)

            agent.store_reward(reward, env.is_terminal())

            episode_reward += reward
            state = next_state

        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

        if (episode + 1) % update_interval == 0:
            agent.update()

        if (episode + 1) % 100 == 0:
            win_rate = wins / (episode + 1) * 100
            loss_rate = losses / (episode + 1) * 100
            draw_rate = draws / (episode + 1) * 100
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Win rate: {win_rate:.1f}%, Loss rate: {loss_rate:.1f}%, Draw rate: {draw_rate:.1f}%")
            print(f"  Total wins: {wins}, losses: {losses}, draws: {draws}")

    print("\n=== Training Complete ===")
    print(f"Final Win rate: {wins/episodes*100:.1f}%")
    print(f"Final Loss rate: {losses/episodes*100:.1f}%")
    print(f"Final Draw rate: {draws/episodes*100:.1f}%")

    return agent


def evaluate_agent(agent, num_games=100):
    env = TicTacToe(seed=123)
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        state = env.reset()

        while not env.is_terminal():
            available_actions = env.get_available_actions()
            if len(available_actions) == 0:
                break

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = agent.policy(state_tensor)

                mask = torch.zeros_like(action_probs)
                mask[0, available_actions] = 1
                action_probs = action_probs * mask
                action_probs = action_probs / action_probs.sum()

                action = torch.argmax(action_probs).item()

            state, reward = env.step(action)

        score = env.get_score()
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1

    print("\n=== Evaluation Results ===")
    print(f"Games played: {num_games}")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")


if __name__ == "__main__":
    print("Training PPO-A2C on Tic-Tac-Toe...")
    agent = train_ppo_tictactoe(episodes=2000, update_interval=20)

    print("\nEvaluating trained agent...")
    evaluate_agent(agent, num_games=100)