import sys
import os
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.double_deep_qlearning_er import DoubleDeepQLearningWithExperienceReplay
from agents.double_deep_qlearning_per import DoubleDeepQLearningWithPrioritizedExperienceReplay
from environnements.test_env.line_world import LineWorld
from environnements.test_env.grid_world import GridWorld
from environnements.test_env.tic_tac_toe import TicTacToe
from evaluate.tracker import RLTracker


EVAL_MILESTONES = [1_000, 10_000, 100_000]
NUM_EVAL_EPISODES = 200
HIDDEN_DIM = 128
LOG_DIR = "runs"
CSV_DIR = "results"

CONFIG = {
    "name": "default",
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.999,
    "buffer_capacity": 10_000,
    "batch_size": 64,
    "target_update_freq": 20,
}

PER_EXTRA = {
    "alpha": 0.6,
    "beta_start": 0.4,
    "beta_frames": 100_000,
}

ENVS = [
    (LineWorld, "LineWorld"),
    (GridWorld, "GridWorld"),
    (TicTacToe, "TicTacToe"),
]


def _get_env_meta(env_class):
    """Détecte dynamiquement state_size et action_size."""
    env = env_class()
    env.reset()
    state = env.get_state()

    if isinstance(state, np.ndarray):
        state_size = state.size
    elif isinstance(state, (list, tuple)):
        state_size = len(state)
    else:
        state_size = 1 #scalaire (LW)

    if hasattr(env, "action_size"):
        action_size = env.action_size
    else:
        all_actions = set()
        for _ in range(50):
            env.reset()
            for a in env.get_available_actions():
                all_actions.add(int(a))
        action_size = max(all_actions) + 1 if all_actions else 4
    return state_size, action_size


# ÉVALUATION POLICY (greedy)
# 1=win, -1=loss, 0=draw

def evaluate_policy(agent, env_class, num_eval_episodes: int):
    wins   = 0
    losses = 0
    draws  = 0
    total_steps     = 0
    total_moves     = 0
    total_move_time = 0.0

    for _ in range(num_eval_episodes):
        env   = env_class()
        state = env.reset()
        episode_steps = 0

        while not env.is_terminal():
            avail = env.get_available_actions()
            if not avail:
                break

            t0     = time.time()
            # select_action retourne un entier (ces agents travaillent en int)
            action = agent.select_action(state, avail, training=False)
            total_move_time += time.time() - t0
            total_moves     += 1

            if action is None:
                break

            state, _ = env.step(action)
            episode_steps += 1

        score = env.get_score() if hasattr(env, "get_score") else 0
        if score == 1:
            wins   += 1
        elif score == -1:
            losses += 1
        else:
            draws  += 1
        total_steps += episode_steps

    avg_score = wins / num_eval_episodes
    loss_rate = losses / num_eval_episodes
    draw_rate = draws / num_eval_episodes
    avg_steps = total_steps / num_eval_episodes
    avg_move_time = total_move_time / total_moves if total_moves > 0 else 0.0
    return avg_score, loss_rate, draw_rate, avg_steps, avg_move_time

def run(agent_class, agent_label: str, config: dict,
        env_class, env_name: str, state_size: int, action_size: int):

    config_name  = config["name"]

    agent_kwargs = {k: v for k, v in config.items() if k != "name"}
    agent_kwargs.update({
        "state_size":  state_size,
        "action_size": action_size,
        "hidden_dim":  HIDDEN_DIM,
    })
    if "PER" in agent_label or "Prioritized" in agent_label:
        agent_kwargs.update(PER_EXTRA)

    agent = agent_class(**agent_kwargs)

    tracker = RLTracker(
        agent_name=f"{agent_label}_{config_name}",
        env_name=env_name,
        config={
            **config,
            "agent": agent_label,
            "hidden_dim": HIDDEN_DIM,
            "eval_milestones": EVAL_MILESTONES,
            "num_eval_episodes": NUM_EVAL_EPISODES,
        },
        log_dir=LOG_DIR,
        csv_dir=CSV_DIR,
    )

    print(f"\n{'='*65}")
    print(f"{agent_label} | {env_name} | {config_name}")
    print(f"Réseau : {state_size} → {HIDDEN_DIM} → ReLU → {HIDDEN_DIM} → ReLU → {action_size}")
    print(f"{'='*65}")

    agent._int_to_action = lambda n: n
    env = env_class()
    cumulative = 0

    try:
        for milestone in EVAL_MILESTONES:
            episodes_needed = milestone - cumulative
            if episodes_needed <= 0:
                continue

            print(
                f"\n[Train] {cumulative:>7,} → {milestone:>7,} "
                f"({episodes_needed:,} épisodes)...",
                end="", flush=True,
            )
            t0 = time.time()
            agent.train(env, num_episodes=episodes_needed)
            print(f"  ✓  {time.time() - t0:.1f}s")

            #eval greedy
            print(
                f"[Eval] {NUM_EVAL_EPISODES} épisodes...",
                end="", flush=True,
            )
            start_total_time = time.time()

            avg_score, loss_rate, draw_rate, avg_steps, avg_move_time = evaluate_policy(
                agent=agent,
                env_class=env_class,
                num_eval_episodes=NUM_EVAL_EPISODES,
            )

            total_time = time.time() - start_total_time

            print(f"[DEBUG] LOGGED STEP = {milestone}")

            tracker.log_evaluation(
                num_episodes_trained=milestone,
                avg_score=avg_score,
                avg_steps=avg_steps,
                loss_rate=loss_rate,
                draw_rate=draw_rate,
                avg_move_time_sec=avg_move_time,
                total_eval_time_sec=total_time,
            )
            tracker.writer.flush()

            print(
                f"[EVAL] Episode {milestone} | "
                f"win_rate={avg_score:.4f} | "
                f"loss_rate={loss_rate:.4f} | "
                f"draw_rate={draw_rate:.4f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"avg_move_time={avg_move_time:.6f}s | "
                f"total_time={total_time:.2f}s"
            )
            cumulative = milestone

    finally:
        tracker.finish()

def main():
    print("=" * 65)
    print("DDQN-ER & DDQN-PER — LineWorld / GridWorld / TicTacToe")
    print(f"Réseau : état → {HIDDEN_DIM} → ReLU → {HIDDEN_DIM} → ReLU → actions")
    print(f"Jalons : {EVAL_MILESTONES}")
    print(f"Éval/jalon : {NUM_EVAL_EPISODES} épisodes")
    print("=" * 65)

    agents_to_test = [
        (DoubleDeepQLearningWithExperienceReplay, "DDQN-ER"),
        (DoubleDeepQLearningWithPrioritizedExperienceReplay, "DDQN-PER"),
    ]

    for env_class, env_name in ENVS:
        state_size, action_size = _get_env_meta(env_class)
        print(f"\n{'▶ ' + env_name + f' (state={state_size}, actions={action_size})':^65}")

        for agent_class, agent_label in agents_to_test:
            run(
                agent_class=agent_class,
                agent_label=agent_label,
                config=CONFIG,
                env_class=env_class,
                env_name=env_name,
                state_size=state_size,
                action_size=action_size,
            )

    print("\n✓ Terminé.")

if __name__ == "__main__":
    main()
