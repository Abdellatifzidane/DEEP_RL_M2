import time

from agents.simulation.randomrollout import RandomRolloutAgent
from environnements.quarto.quatro import Quatro
from evaluate.tracker import RLTracker


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "agent_name": "RandomRollout",
    "env_name": "Quatro",

    "num_rollouts": 10,
    "max_rollout_steps": 1000,

    # mêmes milestones que les autres agents (pour comparaison)
    "eval_milestones": [1_000, 10_000, 100_000, 200_000, 300_000],

    "num_eval_episodes": 100,

    "log_dir": "runs",
    "csv_dir": "results",
}


# ============================================================
# EVALUATION
# ============================================================

def evaluate_agent(agent, env_class, num_eval_episodes):
    total_score = 0.0
    total_steps = 0
    total_moves = 0
    total_move_time = 0.0

    for _ in range(num_eval_episodes):
        env = env_class()
        env.reset()

        episode_steps = 0

        while not env.is_terminal():
            start_time = time.time()

            action = agent.act(env)

            end_time = time.time()

            total_move_time += end_time - start_time
            total_moves += 1

            if action is None:
                break

            env.step(action)
            episode_steps += 1

        total_score += env.get_score()
        total_steps += episode_steps

    avg_score = total_score / num_eval_episodes
    avg_steps = total_steps / num_eval_episodes
    avg_move_time = total_move_time / total_moves if total_moves > 0 else 0.0

    return avg_score, avg_steps, avg_move_time


# ============================================================
# MAIN
# ============================================================

def main():
    agent = RandomRolloutAgent(
        num_rollouts=CONFIG["num_rollouts"],
        max_rollout_steps=CONFIG["max_rollout_steps"],
    )

    tracker = RLTracker(
        agent_name=CONFIG["agent_name"],
        env_name=CONFIG["env_name"],
        config=CONFIG,
        log_dir=CONFIG["log_dir"],
        csv_dir=CONFIG["csv_dir"],
    )

    try:
        for milestone in CONFIG["eval_milestones"]:
            avg_score, avg_steps, avg_move_time = evaluate_agent(
                agent=agent,
                env_class=Quatro,
                num_eval_episodes=CONFIG["num_eval_episodes"],
            )

            tracker.log_evaluation(
                num_episodes_trained=milestone,
                avg_score=avg_score,
                avg_steps=avg_steps,
            )

            tracker.log_move_time(avg_move_time)

            print(
                f"[EVAL] Episode {milestone} | "
                f"avg_score={avg_score:.4f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"avg_move_time={avg_move_time:.6f}s"
            )

    finally:
        tracker.finish()


if __name__ == "__main__":
    main()