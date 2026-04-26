import os
import sys
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.quarto.quatro import Quatro
from agents.mcts import MCTSAgent
from evaluate.tracker import RLTracker


def evaluate_mcts_vs_random(num_simulations=1000, num_games=100):
    """Évalue MCTS (greedy) contre un joueur random."""
    env = Quatro()
    agent = MCTSAgent(num_simulations=num_simulations)

    total_score = 0
    total_steps = 0
    move_times = []

    for game in range(num_games):
        env.reset()
        agent_plays_first = game % 2 == 0
        steps = 0

        while not env.is_terminal():
            available_actions = env.get_available_actions()
            if not available_actions:
                break

            is_agent_turn = (env.current_player == 0 and agent_plays_first) or \
                            (env.current_player == 1 and not agent_plays_first)

            if is_agent_turn:
                t0 = time.time()
                action = agent.choose_action(env)
                move_times.append(time.time() - t0)
            else:
                action = random.choice(available_actions)

            env.step(action)
            steps += 1

        score = env.get_score()
        if score > 0:
            winner = env.current_player
            agent_won = (winner == 0 and agent_plays_first) or \
                        (winner == 1 and not agent_plays_first)
            total_score += 1 if agent_won else -1

        total_steps += steps

    avg_score = total_score / num_games
    avg_steps = total_steps / num_games
    avg_move_time = sum(move_times) / len(move_times) if move_times else 0

    print(f"  [MCTS {num_simulations} sims] score={avg_score:.3f}  "
          f"steps={avg_steps:.1f}  move_time={avg_move_time*1000:.1f} ms")

    return avg_score, avg_steps, avg_move_time


if __name__ == "__main__":
    print("=== MCTS vs Random sur Quarto ===\n")

    sim_configs = [100, 500, 1000]

    for n_sims in sim_configs:
        config = {"num_simulations": n_sims, "num_games": 50}

        tracker = RLTracker(
            agent_name="MCTS",
            env_name="Quarto",
            config=config,
            run_name=f"mcts_{n_sims}sims",
        )

        print(f"--- {n_sims} simulations par coup ---")
        avg_score, avg_steps, avg_move_time = evaluate_mcts_vs_random(
            num_simulations=n_sims, num_games=50
        )

        tracker.log_evaluation(n_sims, avg_score=avg_score, avg_steps=avg_steps)
        tracker.log_move_time(avg_move_time)
        tracker.finish()
        print()

    print("=== DONE ===")
